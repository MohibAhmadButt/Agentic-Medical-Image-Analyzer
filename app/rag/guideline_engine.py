"""
Phase 3: Clinical Guideline Retrieval Engine (ACR Appropriateness Criteria & Clinical Consensus).
Provides verifiable, cited radiological management rules for downstream LLM specialist agents.
"""

from typing import Dict, List


# ============================================================================
# CLINICAL GUIDELINE KNOWLEDGE BASE (ACR APPROPRIATENESS CRITERIA)
# ============================================================================
CLINICAL_GUIDELINES: List[Dict[str, str]] = [
    {
        "compartment": "Pelvis & Hip",
        "condition": "Femoral Neck Fracture / Hip Fracture",
        "guideline_source": "ACR Appropriateness Criteria: Acute Hip Pain - Suspected Fracture (Rev. 2024)",
        "imaging_protocol": "Initial AP Pelvis and dedicated AP/cross-table lateral hip radiographs. If initial radiograph is equivocal but clinical suspicion remains high, non-contrast MRI pelvis/hip is the gold standard modality.",
        "management_rules": "Strict non-weight-bearing precautions immediately. Urgent orthopedic surgery evaluation within 24–48 hours for operative fixation (cannulated screws, sliding hip screw) or arthroplasty (hemiarthroplasty/total hip) to minimize femoral head avascular necrosis (AVN) risk."
    },
    {
        "compartment": "Pelvis & Hip",
        "condition": "Intertrochanteric Femur Fracture",
        "guideline_source": "ACR Appropriateness Criteria: Suspected Extracapsular Hip Trauma",
        "imaging_protocol": "AP Pelvis and cross-table lateral of affected femur. CT pelvis/hip without contrast recommended if comminution or subtrochanteric extension is suspected.",
        "management_rules": "Immobilization, analgesia, and urgent orthopedic consultation for internal fixation via intramedullary nail (cephalomedullary nail) or dynamic hip screw."
    },
    {
        "compartment": "Thorax & Lungs",
        "condition": "Pneumonia / Consolidation",
        "guideline_source": "ACR Appropriateness Criteria: Routine Chest Radiography & Acute Respiratory Illness",
        "imaging_protocol": "Standard Posteroanterior (PA) and Lateral Chest Radiography. If non-resolving after 4–6 weeks or complicated by suspected abscess/empyema, order CT Chest with IV contrast.",
        "management_rules": "Calculate CURB-65 / Pneumonia Severity Index (PSI) score for outpatient vs. inpatient triage. Initiate targeted empiric antimicrobial therapy based on community-acquired vs. hospital-acquired criteria."
    },
    {
        "compartment": "Thorax & Lungs",
        "condition": "Pneumothorax",
        "guideline_source": "ACR Appropriateness Criteria: Thoracic Trauma and Acute Breathlessness",
        "imaging_protocol": "Upright erect PA chest radiograph during inspiration. Expiratory or lateral decubitus views if small apical pneumothorax is suspected.",
        "management_rules": "If tension pneumothorax (tracheal shift, hypotension), immediate emergency needle decompression followed by tube thoracostomy (chest drain). If small (<2cm) and stable, high-flow oxygen and observation."
    },
    {
        "compartment": "Thorax & Lungs",
        "condition": "Pleural Effusion",
        "guideline_source": "ACR Appropriateness Criteria: Evaluation of Pleural Fluid",
        "imaging_protocol": "PA and Lateral Chest Radiograph (blunting requires >150-200 mL on PA, >50 mL on lateral). Thoracic Ultrasound is preferred for bedside quantification and thoracentesis guidance.",
        "management_rules": "Diagnostic thoracentesis with pleural fluid analysis (Light's criteria: protein, LDH, glucose, pH, Gram stain, cytology) to differentiate transudate from exudate."
    },
    {
        "compartment": "Neuro & Cranium",
        "condition": "Acute Intracranial Hemorrhage",
        "guideline_source": "ACR Appropriateness Criteria & AHA/ASA Guidelines for Spontaneous Intracerebral Hemorrhage",
        "imaging_protocol": "Emergency Non-Contrast Head CT (NCCT) is the initial test of choice. Follow with CT Angiography (CTA) head/neck to evaluate for spot sign, vascular malformations, or aneurysm.",
        "management_rules": "Immediate neurosurgical consultation. Tight blood pressure control (target systolic <140 mmHg), rapid reversal of any anticoagulation/coagulopathy, and elevate head of bed 30 degrees to manage intracranial pressure (ICP)."
    },
    {
        "compartment": "Neuro & Cranium",
        "condition": "Acute Ischemic Infarction / Stroke",
        "guideline_source": "AHA/ASA Stroke Guidelines & ACR Cerebrovascular Disease Criteria",
        "imaging_protocol": "Emergency NCCT to rule out hemorrhage, followed immediately by CTA Head/Neck and CT Perfusion (CTP) to map core vs. ischemic penumbra.",
        "management_rules": "Assess eligibility for IV thrombolysis (tissue plasminogen activator / Tenecteplase) within <4.5 hours of symptom onset. Assess for mechanical endovascular thrombectomy (EVT) within 6–24 hours for large vessel occlusions (LVO)."
    },
    {
        "compartment": "Abdomen",
        "condition": "Bowel Obstruction",
        "guideline_source": "ACR Appropriateness Criteria: Suspected Small-Bowel Obstruction",
        "imaging_protocol": "Abdominal Radiograph (KUB / Erect + Supine). CT Abdomen and Pelvis with IV contrast is the definitive imaging modality for transition point and ischemia evaluation.",
        "management_rules": "Nasogastric (NG) tube decompression, IV fluid resuscitation, bowel rest, and urgent surgical consultation if signs of closed-loop obstruction, strangulation, or bowel ischemia are present."
    },
    {
        "compartment": "Skeletal Extremities",
        "condition": "Displaced / Complete Fracture",
        "guideline_source": "ACR Appropriateness Criteria: Acute Trauma to the Extremities",
        "imaging_protocol": "Minimum of 2 orthogonal radiographic views (AP and Lateral). Include joint above and joint below the injured segment.",
        "management_rules": "Neurovascular status check (distal pulses, motor/sensory function). Immediate reduction and splinting if displaced, followed by orthopedic consultation."
    },
    {
        "compartment": "Spine",
        "condition": "Vertebral Compression Fracture",
        "guideline_source": "ACR Appropriateness Criteria: Suspected Spine Trauma",
        "imaging_protocol": "Non-contrast CT spine or sagittal/AP radiographs. Sagittal MRI spine with STIR sequence is recommended to distinguish acute marrow edema from chronic compression.",
        "management_rules": "Neurological examination, pain management, early mobilization with bracing, and consideration for kyphoplasty/vertebroplasty if intractable pain persists."
    }
]


# ============================================================================
# GUIDELINE RETRIEVER CLASS
# ============================================================================
class ClinicalGuidelineRetriever:
    def __init__(self):
        self.guidelines = CLINICAL_GUIDELINES

    def retrieve(self, compartment: str, primary_finding: str) -> Dict[str, str]:
        """Retrieves matching ACR Appropriateness Criteria based on compartment & pathology."""
        # 1. Exact or partial finding match
        finding_lower = primary_finding.lower()
        for g in self.guidelines:
            cond_lower = g["condition"].lower()
            if cond_lower in finding_lower or finding_lower in cond_lower:
                return g

        # 2. Compartment match fallback
        for g in self.guidelines:
            if g["compartment"].lower() in compartment.lower():
                return g

        # 3. Default safe baseline
        return {
            "compartment": compartment,
            "condition": primary_finding,
            "guideline_source": "ACR General Diagnostic Imaging Appropriateness Criteria",
            "imaging_protocol": "Standard 2-view radiographic evaluation or targeted cross-sectional CT/MRI as indicated by clinical progression.",
            "management_rules": "Clinical correlation with physical exam, non-urgent specialist consultation, and repeat imaging if symptoms fail to resolve."
        }


# Singleton instance
guideline_retriever = ClinicalGuidelineRetriever()
