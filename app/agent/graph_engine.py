"""
LangGraph Multi-Agent Clinical Reasoning Engine with Stateful Memory.
Routes BiomedCLIP vision telemetry through specialized radiology protocols using LLaMA 3.3 70B.
"""

import os
from typing import Annotated, Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ============================================================================
# SPECIALIST KNOWLEDGE BASE & RADIOLOGICAL PROTOCOLS
# ============================================================================
MODALITY_PROTOCOLS = {
    "Bone Radiograph / X-Ray": {
        "technique": (
            "Systematically trace cortical outlines from end to end. Evaluate bone density, "
            "trabecular patterns, Shenton's line (for hips/pelvis), joint spaces, and look for sharp steps, "
            "subcapital/cervical angulations, or displaced fracture fragments."
        ),
        "checklist": (
            "Femoral neck fractures, cortical disruption, avulsion fractures, dislocations, "
            "osteopenia/osteoporosis, joint space narrowing, and avascular necrosis risk."
        )
    },
    "Chest X-Ray": {
        "technique": (
            "Use the systematic ABCDE approach: Airway (tracheal alignment), Breathing (bilateral lung fields, "
            "costophrenic angles, infiltrates), Circulation (cardiothoracic ratio < 0.5), Disability (rib/clavicle fractures), "
            "Everything else (diaphragm contours, gastric bubble)."
        ),
        "checklist": (
            "Pneumonia, pulmonary consolidation, pleural effusion, pneumothorax, "
            "pulmonary edema, cardiomegaly, and lung nodules."
        )
    },
    "Brain CT / MRI": {
        "technique": (
            "Evaluate cerebral symmetry. Search for hyperdense areas (acute hemorrhage/blood), "
            "hypodense territories (ischemic infarcts/edema), midline shift, mass effect, and ventricular compression."
        ),
        "checklist": (
            "Acute ischemic stroke, intracranial hemorrhage (subdural/epidural/intracerebral), "
            "mass lesion, cerebral edema, and skull fractures."
        )
    },
    "Dental Panorex": {
        "technique": (
            "Examine enamel-dentin boundaries, pulp chambers, alveolar bone crest levels, "
            "and periapical periodontal ligament spaces across maxillary and mandibular arches."
        ),
        "checklist": (
            "Dental caries, periapical abscesses, impacted third molars, alveolar bone loss, and jaw cysts."
        )
    },
    "Abdominal Scan": {
        "technique": (
            "Assess bowel gas pattern (air-fluid levels, dilated loops), psoas muscle shadows, "
            "calcifications, and check for free peritoneal air under the diaphragm."
        ),
        "checklist": (
            "Bowel obstruction, pneumoperitoneum (perforation), kidney stones, and ascites."
        )
    }
}

# ============================================================================
# STATE DEFINITION & METADATA MERGER
# ============================================================================
def merge_metadata(current: dict, update: dict) -> dict:
    """Safely merges metadata so follow-up chat messages never overwrite prior diagnostic data."""
    if current is None:
        return update if update is not None else {}
    if update is None:
        return current
    merged = current.copy()
    merged.update(update)
    return merged

class ClinicalAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    session_metadata: Annotated[dict, merge_metadata]

# ============================================================================
# GRAPH ENGINE
# ============================================================================
class MedicalGraphEngine:
    def __init__(self, api_key: str = None):
        groq_key = api_key or os.environ.get("GROQ_API_KEY")
        
        # High-performance LLaMA 3.3 70B via Groq
        self.llm = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ClinicalAgentState)

        workflow.add_node("triage_node", self._triage_node)
        workflow.add_node("specialist_node", self._specialist_node)
        workflow.add_node("qa_node", self._qa_node)

        workflow.add_conditional_edges(START, self._router)
        workflow.add_edge("triage_node", "specialist_node")
        workflow.add_edge("specialist_node", "qa_node")
        workflow.add_edge("qa_node", END)

        return workflow.compile(checkpointer=self.memory)

    def _router(self, state: ClinicalAgentState) -> str:
        if state.get("session_metadata", {}).get("report_generated", False):
            return "qa_node"
        return "triage_node"

    def _triage_node(self, state: ClinicalAgentState) -> dict:
        modality_hint = state.get("session_metadata", {}).get("standard_modality", "Bone Radiograph / X-Ray")
        
        selected_modality = "Bone Radiograph / X-Ray"
        for key in MODALITY_PROTOCOLS.keys():
            if key.lower() in modality_hint.lower() or modality_hint.lower() in key.lower():
                selected_modality = key
                break

        return {"session_metadata": {"active_modality": selected_modality}}

    def _specialist_node(self, state: ClinicalAgentState) -> dict:
        modality = state.get("session_metadata", {}).get("active_modality", "Bone Radiograph / X-Ray")
        protocol = MODALITY_PROTOCOLS.get(modality, MODALITY_PROTOCOLS["Bone Radiograph / X-Ray"])
        
        primary_finding = state.get("session_metadata", {}).get("primary_finding", "No primary finding")
        differentials = state.get("session_metadata", {}).get("differentials", "None")
        confidence = state.get("session_metadata", {}).get("modality_confidence", "N/A")

        system_instruction = (
            f"You are a Senior Consulting Radiologist and AI Clinical Decision Support Specialist.\n"
            f"You are analyzing telemetry from Microsoft's BiomedCLIP vision foundation model for a **{modality}**.\n\n"
            f"### BIOMEDICAL VISION FINDINGS:\n"
            f"- Modality: {modality} (Detection Confidence: {confidence}%)\n"
            f"- Primary Pathology Marker: {primary_finding}\n"
            f"- Differential Findings: {differentials}\n\n"
            f"### RADIOLOGICAL PROTOCOL FOR {modality.upper()}:\n"
            f"- Systematic Technique: {protocol['technique']}\n"
            f"- Pathology Checklist: {protocol['checklist']}\n\n"
            f"### INSTRUCTIONS:\n"
            f"Generate a rigorous, professional clinical report strictly following this markdown structure:\n\n"
            f"### 📋 Primary Diagnostic Impression\n"
            f"[Provide a definitive, clear clinical statement of the finding, e.g. Left Femoral Neck Fracture, Transcervical Type]\n\n"
            f"### 🔬 Detailed Observations & Localization\n"
            f"[Detail anatomical landmarks: Cortical continuity, Shenton's line, joint alignment, displacement, and density]\n\n"
            f"### 📊 Differential Diagnoses\n"
            f"- **Primary Suspect:** (Explain reasoning based on visual evidence and risk of avascular necrosis/complications)\n"
            f"- **Secondary Differential:** (Alternative considerations)\n\n"
            f"### ⚠️ Urgency Level\n"
            f"**[Routine | Expedited | Immediate / Emergency Orthopedic/Clinical Evaluation]**\n\n"
            f"### 💡 Recommended Clinical Next Steps\n"
            f"- Immediate stabilization / non-weight-bearing\n"
            f"- Urgent orthopedic consultation (e.g. surgical fixation vs. hemiarthroplasty)\n"
            f"- Confirmatory cross-table lateral imaging or CT\n\n"
            f"---\n*Disclaimer: AI clinical decision-support utility. Requires verification by a certified radiologist/physician.*"
        )

        response = self.llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Synthesize the comprehensive radiology report for this {modality} scan.")
        ])

        return {
            "messages": [response],
            "session_metadata": {
                "report_content": response.content,
                "report_generated": True
            }
        }

    def _qa_node(self, state: ClinicalAgentState) -> dict:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return {}

        report = state.get("session_metadata", {}).get("report_content", "No prior report found.")
        modality = state.get("session_metadata", {}).get("active_modality", "Medical Scan")

        system_prompt = (
            f"You are a clinical specialist discussing an analyzed {modality} with a patient or physician.\n"
            f"OFFICIAL REPORT SUMMARY IN CONTEXT:\n"
            f"--------------------------------------------------\n"
            f"{report}\n"
            f"--------------------------------------------------\n\n"
            f"RULES:\n"
            f"1. You are in interactive chat mode. DO NOT output a new report header.\n"
            f"2. Answer user questions directly, empathetically, and accurately based on the report findings above.\n"
            f"3. Explain treatment implications (e.g. surgery, screws, hip replacement, non-weight-bearing) if asked.\n"
            f"4. Obey formatting constraints strictly (e.g. short summaries, bullet points)."
        )

        clean_history = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                clean_history.append(msg)
            elif isinstance(msg, HumanMessage):
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                clean_history.append(HumanMessage(content=text))

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            *clean_history
        ])

        return {"messages": [response]}

    def invoke_with_memory(self, user_query: str, thread_id: str, extra_meta: dict = None) -> str:
        input_state = {"messages": [HumanMessage(content=str(user_query))]}
        if extra_meta:
            input_state["session_metadata"] = extra_meta

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(input_state, config=config)

        final_msg = result["messages"][-1]
        return final_msg.content if hasattr(final_msg, "content") else "Consultation completed."
