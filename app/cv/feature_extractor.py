"""
BiomedCLIP Zero-Shot Feature Extractor with Visual Attention Heatmaps.
Performs medical zero-shot classification and generates visual localization overlays.
"""

import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import open_clip
from typing import Union, List, Dict, Tuple


class BiomedFeatureExtractor:
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Microsoft BiomedCLIP
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
            "skeletal extremity bone radiograph x-ray",
            "chest radiograph x-ray",
            "brain ct or mri scan",
            "dental panoramic radiograph",
            "abdominal radiograph or ct scan"
        ]

        # Pathologies & Findings
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

    def _load_image_stream(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        return Image.open(image_input).convert("RGB")

    def _generate_heatmap_overlay(self, original_img: Image.Image, image_tensor: torch.Tensor, top_text_token: torch.Tensor) -> Image.Image:
        """Generates an attention attribution heatmap overlaid onto the original medical scan."""
        try:
            # Enable gradient calculation for visual explanation
            image_tensor = image_tensor.clone().detach().requires_grad_(True)
            
            # Forward pass through visual tower
            img_feat = self.model.encode_image(image_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            
            text_feat = self.model.encode_text(top_text_token)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
            # Similarity score for the top detected pathology
            score = (100.0 * img_feat @ text_feat.T).squeeze()
            
            # Backward pass to get input saliency
            self.model.zero_grad()
            score.backward(retain_graph=False)
            
            # Extract gradients and compute spatial importance
            gradients = image_tensor.grad.data.abs().squeeze(0)  # (3, H, W)
            saliency = gradients.mean(dim=0).cpu().numpy()       # (H, W)
            
            # Normalize saliency map [0, 1]
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            # Resize saliency to original scan dimensions
            orig_w, orig_h = original_img.size
            saliency_img = Image.fromarray((saliency * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            saliency_arr = np.array(saliency_img, dtype=np.float32) / 255.0
            
            # Generate Jet colormap (Red = High attention, Blue = Low attention)
            orig_arr = np.array(original_img).astype(np.float32)
            
            # Heatmap color interpolation
            r = np.clip(1.5 - np.abs(saliency_arr * 4 - 3), 0, 1)
            g = np.clip(1.5 - np.abs(saliency_arr * 4 - 2), 0, 1)
            b = np.clip(1.5 - np.abs(saliency_arr * 4 - 1), 0, 1)
            heatmap = np.stack([r, g, b], axis=-1) * 255.0
            
            # Blend original scan with heatmap (65% original, 35% heatmap)
            blended = (0.65 * orig_arr + 0.35 * heatmap).clip(0, 255).astype(np.uint8)
            return Image.fromarray(blended)
        except Exception:
            # Safe fallback: return original image if gradient hook fails
            return original_img

    def extract_features(self, image_input: Union[str, bytes, Image.Image]) -> Tuple[Dict, Image.Image]:
        """Runs zero-shot classification and generates visual attention heatmap."""
        image = self._load_image_stream(image_input)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        mod_tokens = self.tokenizer(
            [f"this is a medical scan showing {m}" for m in self.modality_labels]
        ).to(self.device)

        path_tokens = self.tokenizer(
            [f"a radiograph demonstrating {p}" for p in self.pathology_labels]
        ).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            mod_feat = self.model.encode_text(mod_tokens)
            mod_feat /= mod_feat.norm(dim=-1, keepdim=True)
            mod_probs = (100.0 * img_feat @ mod_feat.T).softmax(dim=-1).cpu().numpy()[0]

            path_feat = self.model.encode_text(path_tokens)
            path_feat /= path_feat.norm(dim=-1, keepdim=True)
            path_probs = (100.0 * img_feat @ path_feat.T).softmax(dim=-1).cpu().numpy()[0]

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

        # Generate Attention Heatmap for Top Finding
        top_idx = int(path_probs.argmax())
        top_token = path_tokens[top_idx : top_idx + 1]
        heatmap_img = self._generate_heatmap_overlay(image, image_tensor, top_token)

        telemetry = {
            "standard_modality": standard_modality,
            "modality_confidence": round(float(mod_probs.max()) * 100, 2),
            "primary_finding": ranked_pathologies[0],
            "differential_findings": ranked_pathologies[1:4]
        }

        return telemetry, heatmap_img


# Global singleton instance
cv_extractor = BiomedFeatureExtractor()
