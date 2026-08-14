"""
LangGraph Multi-Agent Clinical Triage Engine with Stateful Memory.
Routes telemetry through structured specialist nodes and maintains conversation state.
"""

import os
from typing import Annotated, Dict, Any, List, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ============================================================================
# CLINICAL RADIOLOGY GUIDELINES
# ============================================================================
MODALITY_PROTOCOLS = {
    "Chest X-Ray": {
        "focus": "Inspect lung fields for opacities, cardiomegaly, pleural effusions, and pneumothorax.",
        "checklist": "Pneumonia, Effusion, Infiltrates, Cardiomegaly, Fractures."
    },
    "Bone Radiograph / X-Ray": {
        "focus": "Trace cortical margins for discontinuities, dislocations, and bone density variations.",
        "checklist": "Cortical disruption, Joint space narrowing, Dislocation, Osteolysis."
    },
    "Brain CT / MRI": {
        "focus": "Evaluate midline shift, symmetry, hyperdense acute hemorrhage, or hypodense ischemic infarcts.",
        "checklist": "Mass effect, Ischemic stroke, Subdural/Epidural bleed, Edema."
    },
    "Dental Panorex": {
        "focus": "Examine enamel integrity, alveolar bone levels, and apical radiolucencies.",
        "checklist": "Periapical lesion, Deep caries, Impaction, Periodontal bone loss."
    },
    "Abdominal Scan": {
        "focus": "Assess bowel gas distribution, free peritoneal air, and organ borders.",
        "checklist": "Obstruction, Free fluid, Calcifications, Perforation."
    }
}

# ============================================================================
# STATE DEFINITION & PERSISTENCE REDUCER
# ============================================================================
def merge_metadata(current: dict, update: dict) -> dict:
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
# ENGINE PIPELINE
# ============================================================================
class MedicalGraphEngine:
    def __init__(self, api_key: str = None):
        groq_api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        # High-performance LLaMA 3.3 70B reasoning engine
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ClinicalAgentState)

        builder.add_node("triage_node", self._triage_node)
        builder.add_node("specialist_node", self._specialist_node)
        builder.add_node("qa_node", self._qa_node)

        builder.add_conditional_edges(START, self._router)
        builder.add_edge("triage_node", "specialist_node")
        builder.add_edge("specialist_node", "qa_node")
        builder.add_edge("qa_node", END)

        return builder.compile(checkpointer=self.memory)

    def _router(self, state: ClinicalAgentState) -> str:
        if state.get("session_metadata", {}).get("report_generated", False):
            return "qa_node"
        return "triage_node"

    def _triage_node(self, state: ClinicalAgentState) -> dict:
        input_modality = state.get("session_metadata", {}).get("modality_hint", "Chest X-Ray")
        
        # Match against designated protocols
        selected_modality = "Chest X-Ray"
        for key in MODALITY_PROTOCOLS:
            if key.lower() in input_modality.lower() or input_modality.lower() in key.lower():
                selected_modality = key
                break

        return {"session_metadata": {"active_modality": selected_modality}}

    def _specialist_node(self, state: ClinicalAgentState) -> dict:
        modality = state.get("session_metadata", {}).get("active_modality", "Chest X-Ray")
        protocol = MODALITY_PROTOCOLS.get(modality, MODALITY_PROTOCOLS["Chest X-Ray"])
        visual_findings = state.get("session_metadata", {}).get("findings_summary", "Standard density screening.")

        system_instruction = (
            f"You are a Senior Radiologist and Clinical Decision Support AI specializing in {modality}.\n"
            f"CLINICAL PROTOCOL:\n"
            f"- Primary Focus: {protocol['focus']}\n"
            f"- Diagnostic Checklist: {protocol['checklist']}\n"
            f"- Scan Biomarker Observations: {visual_findings}\n\n"
            f"Generate a rigorous structured report with the following markdown format:\n"
            f"### 📋 Primary Diagnostic Impression\n"
            f"[Direct diagnostic impression detailing likely pathology or clear status]\n\n"
            f"### 🔬 Detailed Observations & Localization\n"
            f"[Systematic breakdown of anatomical structures and visual densities]\n\n"
            f"### 📊 Differential Diagnoses\n"
            f"- Primary Suspect (with clinical justification)\n"
            f"- Secondary Differential\n\n"
            f"### ⚠️ Urgency Level\n"
            f"**[Routine | Expedited | Immediate Emergency]**\n\n"
            f"### 💡 Recommended Next Steps\n"
            f"[Actionable clinical steps, confirmatory imaging, or lab tests]\n\n"
            f"---\n*Disclaimer: AI decision-support utility. All findings require clinical confirmation.*"
        )

        response = self.llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=f"Synthesize the complete radiological consultation report for this {modality} scan.")
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
            f"You are a clinical advisory specialist discussing a previously analyzed {modality} scan.\n"
            f"REPORT IN CONTEXT:\n----------------\n{report}\n----------------\n\n"
            f"INSTRUCTIONS:\n"
            f"1. You are in interactive chat mode. Do not generate a new report template.\n"
            f"2. Answer user questions directly, empathetically, and accurately based on the report findings.\n"
            f"3. Strictly obey user formatting constraints (e.g. short summary, single sentence, bullet points)."
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
