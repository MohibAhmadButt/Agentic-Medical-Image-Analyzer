"""BiomedCLIP Feature Extractor for Zero-Shot Medical Vision Triage."""

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

    # Load Microsoft BiomedCLIP from Hugging Face Hub
    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    self.tokenizer = open_clip.get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    self.model.to(self.device)
    self.model.eval()

    self.modalities = [
        "chest x-ray",
        "brain mri",
        "ct scan",
        "bone radiograph / x-ray",
        "dental panoramic radiograph",
        "histopathology slide",
    ]

    self.pathology_candidates = [
        "normal healthy scan with no obvious abnormality",
        "pneumonia or consolidation",
        "pleural effusion",
        "pulmonary edema",
        "cardiomegaly / enlarged cardiac silhouette",
        "bone fracture or cortical disruption",
        "intracranial hemorrhage or mass lesion",
        "acute ischemic stroke / cerebral infarction",
        "dental caries or periapical lesion",
        "degenerative joint disease or osteoarthritis",
    ]

  def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
    if isinstance(image_input, bytes):
      return Image.open(io.BytesIO(image_input)).convert("RGB")
    return Image.open(image_input).convert("RGB")

  def analyze_image(
      self,
      image_input: Union[str, bytes],
      custom_labels: List[str] = None,
  ) -> Dict:
    image = self._load_image(image_input)
    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

    # Modality Classification
    modality_tokens = self.tokenizer(
        [f"a medical image of {m}" for m in self.modalities]
    ).to(self.device)

    # Pathology Classification
    labels_to_test = custom_labels or self.pathology_candidates
    pathology_tokens = self.tokenizer(
        [f"medical scan demonstrating {p}" for p in labels_to_test]
    ).to(self.device)

    with torch.no_grad():
      image_feat = self.model.encode_image(image_tensor)
      image_feat /= image_feat.norm(dim=-1, keepdim=True)

      # Evaluate Modalities
      mod_feat = self.model.encode_text(modality_tokens)
      mod_feat /= mod_feat.norm(dim=-1, keepdim=True)
      mod_probs = (
          (100.0 * image_feat @ mod_feat.T).softmax(dim=-1).cpu().numpy()[0]
      )
      detected_modality = self.modalities[mod_probs.argmax()]

      # Evaluate Pathologies
      path_feat = self.model.encode_text(pathology_tokens)
      path_feat /= path_feat.norm(dim=-1, keepdim=True)
      path_probs = (
          (100.0 * image_feat @ path_feat.T).softmax(dim=-1).cpu().numpy()[0]
      )

    ranked_pathologies = sorted(
        [
            {"finding": label, "confidence": round(float(prob) * 100, 2)}
            for label, prob in zip(labels_to_test, path_probs)
        ],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    return {
        "detected_modality": detected_modality,
        "modality_confidence": round(float(mod_probs.max()) * 100, 2),
        "primary_finding": ranked_pathologies[0],
        "differential_findings": ranked_pathologies[:4],
    }


# Singleton instance
cv_extractor = BiomedFeatureExtractor()
