"""
Phase 1: BiomedCLIP with Two-Tier Hierarchical Triage & Clinical Prompt Ensembling.
Improves zero-shot classification robustness and anatomical localization.
"""

import io
import torch
import numpy as np
from PIL import Image
import open_clip
from typing import Union, Dict, Tuple, List


# ============================================================================
# TIER 1: ANATOMICAL COMPARTMENTS & PROMPT ENSEMBLES
# ============================================================================
ANATOMY_ENSEMBLE = {
    "Pelvis & Hip": [
        "pelvis or hip bone radiograph x-ray",
        "anteroposterior radiograph of the pelvis and proximal femur",
        "orthopedic radiograph of the hip joint and femoral head",
    ],
    "Skeletal Extremities": [
        "extremity or limb skeletal bone radiograph x-ray",
        "orthopedic bone radiograph of the arm, wrist, leg, or knee",
        "radiograph of long bones demonstrating cortex and trabeculae",
    ],
    "Thorax & Lungs": [
        "chest radiograph x-ray of the lungs and heart",
        "posteroanterior PA or AP chest radiograph",
        "radiograph of the thoracic cavity, lung fields, and mediastinum",
    ],
    "Neuro & Cranium": [
        "brain CT scan or cranial MRI",
        "axial brain computed tomography scan",
        "cross-sectional neuroimaging scan of the head and cranium",
    ],
    "Spine": [
        "cervical, thoracic, or lumbar spine radiograph",
        "vertebral column x-ray demonstrating vertebral bodies and disc spaces",
        "sagittal spinal radiograph",
    ],
    "Dental Panorex": [
        "panoramic dental radiograph orthopantomogram",
        "panorex x-ray of maxillary and mandibular teeth and jaw",
        "dental radiograph of the dentition and alveolar bone",
    ],
    "Abdomen": [
        "abdominal radiograph KUB or CT scan",
        "supine or erect abdominal radiograph showing bowel gas patterns",
        "abdominal imaging demonstrating peritoneal cavity and solid organs",
    ],
}


# ============================================================================
# TIER 2: SPECIALIZED PATHOLOGY ENSEMBLES PER COMPARTMENT
# ============================================================================
PATHOLOGY_ENSEMBLE = {
    "Pelvis & Hip": {
        "Femoral Neck Fracture / Hip Fracture": [
            "radiograph demonstrating subcapital or transcervical femoral neck fracture",
            "impacted intracapsular hip fracture with disruption of Shenton's line",
            "cortical step-off and fracture line across the proximal femoral neck",
            "displaced or non-displaced fracture of the left or right femur neck",
        ],
        "Intertrochanteric Femur Fracture": [
            "extracapsular intertrochanteric fracture between greater and lesser trochanters",
            "comminuted fracture across the trochanteric line of the femur",
        ],
        "Hip Osteoarthritis / Degenerative Joint Disease": [
            "osteoarthritis of the hip with joint space narrowing and subchondral sclerosis",
            "severe degenerative changes of the acetabulofemoral joint with osteophytes",
        ],
        "Avascular Necrosis of Femoral Head": [
            "avascular necrosis showing subchondral collapse and crescent sign in femoral head",
            "femoral head osteonecrosis with patchy sclerosis and flattening",
        ],
        "Normal Hip Alignment (No Fracture)": [
            "normal healthy hip joint with intact cortex and smooth Shenton's line",
            "intact femoral head and neck with no evidence of fracture or dislocation",
        ],
    },
    "Thorax & Lungs": {
        "Pneumonia / Consolidation": [
            "chest radiograph showing focal consolidation or air bronchograms",
            "pulmonary opacity, infiltrate, or alveolar consolidation typical of pneumonia",
        ],
        "Pleural Effusion": [
            "blunting of the costophrenic angle indicating pleural fluid accumulation",
            "moderate to large pleural effusion with meniscus sign",
        ],
        "Pneumothorax": [
            "visible visceral pleural line with absent peripheral lung markings indicating pneumothorax",
            "apical pneumothorax or air in the pleural space",
        ],
        "Cardiomegaly": [
            "enlarged cardiac silhouette with cardiothoracic ratio exceeding 50%",
            "significant cardiomegaly with widened cardiac borders",
        ],
        "Normal Clear Lungs": [
            "clear lung fields with no consolidation, effusion, or pneumothorax",
            "normal healthy chest radiograph with sharp costophrenic angles",
        ],
    },
    "Skeletal Extremities": {
        "Displaced / Complete Fracture": [
            "cortical bone fracture with displacement, angulation, or fragmented cortex",
            "acute traumatic bone fracture with clear cortical step-off",
        ],
        "Buckle / Torus / Incomplete Fracture": [
            "subtle buckle or torus fracture with minor cortical disruption",
            "hairline non-displaced stress fracture",
        ],
        "Normal Bone Alignment": [
            "intact continuous bone cortex with no fracture or dislocation",
            "normal skeletal radiograph without acute osseous abnormality",
        ],
    },
    "Neuro & Cranium": {
        "Acute Intracranial Hemorrhage": [
            "hyperdense acute blood collection, subdural or epidural hematoma on CT",
            "intracerebral hemorrhage with surrounding edema and mass effect",
        ],
        "Acute Ischemic Infarction / Stroke": [
            "hypodense area of cerebral infarction with loss of gray-white differentiation",
            "acute territorial ischemic stroke on non-contrast CT",
        ],
        "Normal Brain Scan": [
            "normal brain parenchyma with symmetric ventricles and no hemorrhage or mass",
            "unremarkable non-contrast brain computed tomography",
        ],
    },
    "Dental Panorex": {
        "Dental Caries / Periapical Lesion": [
            "periapical radiolucency or deep dental caries affecting enamel and dentin",
            "apical periodontitis or tooth decay with bone loss",
        ],
        "Impacted Third Molar": [
            "impacted wisdom tooth / third molar with angulation against second molar",
        ],
        "Normal Dentition": [
            "healthy teeth with intact enamel and normal alveolar crest bone levels",
        ],
    },
    "Abdomen": {
        "Bowel Obstruction": [
            "dilated small bowel loops with multiple air-fluid levels on upright radiograph",
            "mechanical bowel obstruction with plicae circulares stretching",
        ],
        "Pneumoperitoneum / Free Air": [
            "free subdiaphragmatic air crescent indicating hollow viscus perforation",
        ],
        "Normal Abdomen": [
            "non-specific bowel gas pattern with no dilatation or free peritoneal air",
        ],
    },
    "Spine": {
        "Vertebral Compression Fracture": [
            "vertebral body height loss and anterior wedge compression fracture",
            "acute osteoporotic vertebral compression fracture",
        ],
        "Degenerative Disc Disease": [
            "disc space narrowing, osteophyte formation, and endplate sclerosis",
        ],
        "Normal Spine Alignment": [
            "preserved vertebral heights and intact lordotic/kyphotic alignment",
        ],
    },
}


# ============================================================================
# FEATURE EXTRACTOR BACKBONE
# ============================================================================
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

    def _load_image_stream(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        return Image.open(image_input).convert("RGB")

    def _encode_prompt_ensemble(self, prompt_list: List[str]) -> torch.Tensor:
        """Computes the mean normalized text embedding for a list of clinical prompt variants."""
        tokens = self.tokenizer(prompt_list).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            # Average embeddings across ensemble variants and re-normalize
            ensemble_mean = text_embeddings.mean(dim=0, keepdim=True)
            ensemble_mean = ensemble_mean / ensemble_mean.norm(dim=-1, keepdim=True)
        return ensemble_mean

    def _generate_heatmap_overlay(self, original_img: Image.Image, image_tensor: torch.Tensor, text_feat: torch.Tensor) -> Image.Image:
        """Computes input saliency attribution map for the top predicted clinical ensemble."""
        try:
            image_tensor = image_tensor.clone().detach().requires_grad_(True)
            
            img_feat = self.model.encode_image(image_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            
            score = (100.0 * img_feat @ text_feat.T).squeeze()
            
            self.model.zero_grad()
            score.backward(retain_graph=False)
            
            gradients = image_tensor.grad.data.abs().squeeze(0)  # Shape: (3, H, W)
            saliency = gradients.mean(dim=0).cpu().numpy()       # Shape: (H, W)
            
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            
            orig_w, orig_h = original_img.size
            saliency_img = Image.fromarray((saliency * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            saliency_arr = np.array(saliency_img, dtype=np.float32) / 255.0
            
            orig_arr = np.array(original_img).astype(np.float32)
            
            # Color map interpolation (Jet approximation)
            r = np.clip(1.5 - np.abs(saliency_arr * 4 - 3), 0, 1)
            g = np.clip(1.5 - np.abs(saliency_arr * 4 - 2), 0, 1)
            b = np.clip(1.5 - np.abs(saliency_arr * 4 - 1), 0, 1)
            heatmap = np.stack([r, g, b], axis=-1) * 255.0
            
            blended = (0.65 * orig_arr + 0.35 * heatmap).clip(0, 255).astype(np.uint8)
            return Image.fromarray(blended)
        except Exception:
            return original_img

    def extract_features(self, image_input: Union[str, bytes, Image.Image]) -> Tuple[Dict, Image.Image]:
        """Executes Tier-1 Anatomical Triage followed by Tier-2 Pathology Ensembling."""
        image = self._load_image_stream(image_input)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.model.encode_image(image_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # -------------------------------------------------------------
        # TIER 1: ANATOMICAL COMPARTMENT TRIAGE
        # -------------------------------------------------------------
        compartment_names = list(ANATOMY_ENSEMBLE.keys())
        compartment_feats = torch.cat([
            self._encode_prompt_ensemble(ANATOMY_ENSEMBLE[comp])
            for comp in compartment_names
        ], dim=0)

        mod_probs = (100.0 * img_feat @ compartment_feats.T).softmax(dim=-1).cpu().numpy()[0]
        top_comp_idx = int(mod_probs.argmax())
        selected_compartment = compartment_names[top_comp_idx]
        modality_conf = round(float(mod_probs.max()) * 100, 2)

        # -------------------------------------------------------------
        # TIER 2: SPECIALIZED PATHOLOGY ENSEMBLE EVALUATION
        # -------------------------------------------------------------
        pathologies_dict = PATHOLOGY_ENSEMBLE.get(selected_compartment, PATHOLOGY_ENSEMBLE["Pelvis & Hip"])
        pathology_names = list(pathologies_dict.keys())

        pathology_feats = torch.cat([
            self._encode_prompt_ensemble(pathologies_dict[p_name])
            for p_name in pathology_names
        ], dim=0)

        path_probs = (100.0 * img_feat @ pathology_feats.T).softmax(dim=-1).cpu().numpy()[0]

        ranked_pathologies = sorted(
            [
                {"finding": p_name, "confidence": round(float(prob) * 100, 2)}
                for p_name, prob in zip(pathology_names, path_probs)
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )

        # Standard Modality Normalization for LangGraph Protocols
        if "Hip" in selected_compartment or "Skeletal" in selected_compartment or "Spine" in selected_compartment:
            standard_modality = "Bone Radiograph / X-Ray"
        elif "Thorax" in selected_compartment:
            standard_modality = "Chest X-Ray"
        elif "Neuro" in selected_compartment:
            standard_modality = "Brain CT / MRI"
        elif "Dental" in selected_compartment:
            standard_modality = "Dental Panorex"
        else:
            standard_modality = "Abdominal Scan"

        # Generate Saliency Heatmap for the top ranked pathology ensemble
        top_path_feat = pathology_feats[pathology_names.index(ranked_pathologies[0]["finding"]) : pathology_names.index(ranked_pathologies[0]["finding"]) + 1]
        heatmap_img = self._generate_heatmap_overlay(image, image_tensor, top_path_feat)

        telemetry = {
            "anatomical_compartment": selected_compartment,
            "standard_modality": standard_modality,
            "modality_confidence": modality_conf,
            "primary_finding": ranked_pathologies[0],
            "differential_findings": ranked_pathologies[1:],
        }

        return telemetry, heatmap_img


# Singleton instance
cv_extractor = BiomedFeatureExtractor()
