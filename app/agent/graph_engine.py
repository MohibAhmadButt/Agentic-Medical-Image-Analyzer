"""
LangGraph Multi-Agent Medical State Engine with Stateful Memory.
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

from app.llm.report_generator import ClinicalReportSynthesizer, MedicalReport

load_dotenv()

# Specialized Protocols
SPECIALIST_PROTOCOLS = {
    "Bone Radiograph / X-Ray": {
        "technique": "Trace cortical bone contours, inspect Shenton's line, joint alignment, and trabeculae.",
        "checklist": "Femoral neck fractures, cortical disruption, joint space narrowing, and avascular necrosis risk."
    },
    "Chest X-Ray": {
        "technique": "Systematic ABCDE approach (Airway, Breathing, Circulation, Disability, Everything else).",
        "checklist": "Pneumonia, pleural effusion, pneumothorax, pulmonary edema, cardiomegaly."
    },
    "Brain CT / MRI": {
        "technique": "Symmetry evaluation, hyperdense acute blood, hypodense ischemia, midline shift.",
        "checklist": "Acute ischemic stroke, intracranial hemorrhage, mass effect, edema."
    },
    "Dental Panorex": {
        "technique": "Enamel cap, pulp chamber, alveolar crest levels, periapical radiolucencies.",
        "checklist": "Caries, periapical abscess, impaction, periodontal bone loss."
    },
    "Abdominal Scan": {
        "technique": "Bowel gas patterns, air-fluid levels, psoas lines, free air under diaphragm.",
        "checklist": "Bowel obstruction, perforation, calcifications, ascites."
    }
}


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


class MedicalGraphEngine:
    def __init__(self, api_key: str = None):
        self.groq_key = api_key or os.getenv("GROQ_API_KEY")
        self.text_llm = ChatGroq(
            api_key=self.groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )
        self.synthesizer = ClinicalReportSynthesizer(api_key=self.groq_key)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ClinicalAgentState)

        workflow.add_node("triage_agent", self._triage_node)
        workflow.add_node("specialist_agent", self._specialist_node)
        workflow.add_node("qa_agent", self._qa_node)

        workflow.add_conditional_edges(START, self._router)
        workflow.add_edge("triage_agent", "specialist_agent")
        workflow.add_edge("specialist_agent", "qa_agent")
        workflow.add_edge("qa_agent", END)

        return workflow.compile(checkpointer=self.memory)

    def _router(self, state: ClinicalAgentState) -> str:
        if state.get("session_metadata", {}).get("report_generated", False):
            return "qa_agent"
        return "triage_agent"

    def _triage_node(self, state: ClinicalAgentState) -> dict:
        telemetry = state.get("session_metadata", {}).get("telemetry", {})
        modality = telemetry.get("standard_modality", "Bone Radiograph / X-Ray")
        return {"session_metadata": {"active_modality": modality}}

    def _specialist_node(self, state: ClinicalAgentState) -> dict:
        modality = state.get("session_metadata", {}).get("active_modality", "Bone Radiograph / X-Ray")
        protocol = SPECIALIST_PROTOCOLS.get(modality, SPECIALIST_PROTOCOLS["Bone Radiograph / X-Ray"])
        telemetry = state.get("session_metadata", {}).get("telemetry", {})

        report: MedicalReport = self.synthesizer.generate(
            modality=modality,
            telemetry=telemetry,
            protocol=protocol
        )

        formatted_markdown = (
            f"### 📋 Primary Diagnostic Impression\n{report.primary_impression}\n\n"
            f"### 🔬 Key Findings & Localization\n" + "\n".join([f"- {f}" for f in report.key_findings]) + "\n\n"
            f"### 📊 Differential Diagnoses\n" + "\n".join([f"- {d}" for d in report.differential_diagnosis]) + "\n\n"
            f"### ⚠️ Urgency Level\n**{report.urgency_level}** (Confidence: {report.confidence_score}%)\n\n"
            f"### 💡 Clinical Recommendations\n" + "\n".join([f"- {r}" for r in report.recommendations]) + "\n\n"
            f"---\n*{report.disclaimer}*"
        )

        return {
            "messages": [AIMessage(content=formatted_markdown)],
            "session_metadata": {
                "report_structured": report.model_dump(),
                "report_markdown": formatted_markdown,
                "report_generated": True
            }
        }

    def _qa_node(self, state: ClinicalAgentState) -> dict:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            return {}

        report_md = state.get("session_metadata", {}).get("report_markdown", "No report available.")
        modality = state.get("session_metadata", {}).get("active_modality", "Medical Scan")

        system_prompt = (
            f"You are a clinical advisory specialist discussing an analyzed {modality}.\n"
            f"REPORT IN CONTEXT:\n{report_md}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer follow-up questions accurately and concisely based strictly on the report findings above.\n"
            f"2. Explain surgical interventions, non-weight-bearing protocols, or imaging steps if asked.\n"
            f"3. Strictly obey user formatting constraints."
        )

        clean_history = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                clean_history.append(msg)
            elif isinstance(msg, HumanMessage):
                clean_history.append(HumanMessage(content=str(msg.content)))

        response = self.text_llm.invoke([
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
