"""
BiomedCLIP Zero-Shot Feature Extractor
Performs medical zero-shot classification across imaging modalities and pathologies.
"""

import io
import torch
from PIL import Image
import open_clip
from typing import Union, List, Dict


class BiomedFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Microsoft BiomedCLIP from Hugging Face Hub
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        self.model.to(self.device)
        self.model.eval()

        # Clinical Modalities
        self.modality_labels = [
            "pelvis or hip bone radiograph x-ray",
            "extremity or skeletal bone radiograph x-ray",
            "chest radiograph x-ray",
            "brain ct or mri scan",
            "dental panoramic radiograph",
            "abdominal radiograph or ct scan"
        ]

        # Granular Pathologies & Findings
        self.pathology_labels = [
            "femoral neck fracture or hip bone fracture",
            "cortical bone fracture or displaced bone fragment",
            "normal bone alignment with no visible fracture",
            "pneumonia, pulmonary consolidation or infiltrates",
            "pleural effusion or fluid in pleural cavity",
            "normal clear chest radiograph with healthy lungs",
            "acute ischemic cerebral infarction or stroke",
            "intracranial mass lesion or hemorrhage",
            "dental caries or periapical radiolucency",
            "bowel obstruction or abnormal abdominal air"
        ]

    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        return Image.open(image_input).convert("RGB")

    def analyze_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict:
        image = self._load_image(image_input)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # 1. Modality Tokenization
        mod_tokens = self.tokenizer(
            [f"this is a medical scan showing {m}" for m in self.modality_labels]
        ).to(self.device)

        # 2. Pathology Tokenization
        path_tokens = self.tokenizer(
            [f"a radiograph demonstrating {p}" for p in self.pathology_labels]
        ).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            # Modality Logits
            mod_feat = self.model.encode_text(mod_tokens)
            mod_feat /= mod_feat.norm(dim=-1, keepdim=True)
            mod_probs = (100.0 * img_feat @ mod_feat.T).softmax(dim=-1).cpu().numpy()[0]

            # Pathology Logits
            path_feat = self.model.encode_text(path_tokens)
            path_feat /= path_feat.norm(dim=-1, keepdim=True)
            path_probs = (100.0 * img_feat @ path_feat.T).softmax(dim=-1).cpu().numpy()[0]

        # Standardize Modality Mapping
        top_mod_idx = int(mod_probs.argmax())
        raw_modality = self.modality_labels[top_mod_idx]

        if "hip" in raw_modality or "pelvis" in raw_modality or "bone" in raw_modality:
            standard_modality = "Bone Radiograph / X-Ray"
        elif "chest" in raw_modality:
            standard_modality = "Chest X-Ray"
        elif "brain" in raw_modality:
            standard_modality = "Brain CT / MRI"
        elif "dental" in raw_modality:
            standard_modality = "Dental Panorex"
        else:
            standard_modality = "Abdominal Scan"

        ranked_pathologies = sorted(
            [
                {"finding": label, "confidence": round(float(prob) * 100, 2)}
                for label, prob in zip(self.pathology_labels, path_probs)
            ],
            key=lambda x: x["confidence"],
            reverse=True
        )

        return {
            "standard_modality": standard_modality,
            "raw_detected_modality": raw_modality,
            "modality_confidence": round(float(mod_probs.max()) * 100, 2),
            "primary_finding": ranked_pathologies[0],
            "differential_findings": ranked_pathologies[1:4]
        }


# Singleton instance for tool and UI usage
cv_extractor = BiomedFeatureExtractor()
