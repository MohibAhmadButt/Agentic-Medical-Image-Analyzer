import io
from typing import Dict, List, Union
import open_clip
from PIL import Image
import torch


class BiomedFeatureExtractor:

  def __init__(self, device: str = None):
    self.device = (
        device
        if device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    self.tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    self.model.to(self.device)
    self.model.eval()

    # Modality detection labels
    self.modalities = [
        "pelvis or hip bone radiograph x-ray",
        "bone x-ray fracture or extremity",
        "chest radiograph x-ray",
        "brain ct or mri scan",
        "dental panoramic x-ray",
        "abdominal radiograph",
    ]

    # Pathology candidates
    self.pathologies = [
        "femoral neck fracture / hip fracture",
        "cortical bone fracture or disruption",
        "normal bone alignment with no fracture",
        "pulmonary pneumonia or opacity",
        "normal clear chest radiograph",
        "acute ischemic stroke or mass",
        "dental caries or periapical lesion",
    ]

  def analyze_image(self, image: Image.Image) -> Dict:
    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

    # 1. Modality evaluation
    mod_tokens = self.tokenizer(
        [f"this is a medical {m}" for m in self.modalities]
    ).to(self.device)
    path_tokens = self.tokenizer(
        [f"a medical radiograph showing {p}" for p in self.pathologies]
    ).to(self.device)

    with torch.no_grad():
      img_feat = self.model.encode_image(image_tensor)
      img_feat /= img_feat.norm(dim=-1, keepdim=True)

      # Modality scores
      mod_feat = self.model.encode_text(mod_tokens)
      mod_feat /= mod_feat.norm(dim=-1, keepdim=True)
      mod_probs = (100.0 * img_feat @ mod_feat.T).softmax(dim=-1).cpu().numpy()[0]

      # Pathology scores
      path_feat = self.model.encode_text(path_tokens)
      path_feat /= path_feat.norm(dim=-1, keepdim=True)
      path_probs = (
          (100.0 * img_feat @ path_feat.T).softmax(dim=-1).cpu().numpy()[0]
      )

    top_mod_idx = mod_probs.argmax()
    detected_raw = self.modalities[top_mod_idx]

    # Map to standardized category
    if "hip" in detected_raw or "bone" in detected_raw:
      standard_modality = "Bone Radiograph / X-Ray"
    elif "chest" in detected_raw:
      standard_modality = "Chest X-Ray"
    elif "brain" in detected_raw:
      standard_modality = "Brain CT / MRI"
    elif "dental" in detected_raw:
      standard_modality = "Dental Panorex"
    else:
      standard_modality = "Abdominal Scan"

    ranked_path = sorted(
        [
            {"finding": p, "confidence": round(float(prob) * 100, 2)}
            for p, prob in zip(self.pathologies, path_probs)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "modality": standard_modality,
        "modality_confidence": round(float(mod_probs.max()) * 100, 2),
        "primary_pathology": ranked_path[0],
        "differentials": ranked_path[1:4],
    }


cv_extractor = BiomedFeatureExtractor()
