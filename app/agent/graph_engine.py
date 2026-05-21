"""
LangGraph Multi-Agent Triage Engine with Memory
Routes images from a Triage Agent to a Specialist Agent, saving context for a QA Agent.
"""

import os
from typing import Annotated, Literal, Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# ============================================================================
# SPECIALIST KNOWLEDGE BASE
# ============================================================================
DISEASE_CHECKLISTS = {
    "Chest X-Ray": "Analyze the lungs and heart. Check specifically for: Pneumonia, Tuberculosis, Lung cancer, COVID-19 pneumonia, Pleural effusion, Pneumothorax, COPD, Cardiomegaly, and Pulmonary edema.",
    "Brain CT": "Analyze the cranial cavity. Check specifically for: Stroke signs, Brain bleed (hemorrhage), Tumors, Trauma/skull fractures, and Hydrocephalus.",
    "Bone X-Ray": "Analyze the skeletal structures. Check specifically for: Fractures, Osteoporosis signs, Bone tumors, Arthritis (joint space narrowing), and Osteomyelitis.",
    "Abdominal X-Ray": "Check specifically for: Bowel obstruction, Kidney stones, Perforation, and Constipation.",
    "Dental X-Ray": "Check specifically for: Cavities, Impacted teeth, Jaw infection, and Bone loss.",
    "Chest CT": "Check specifically for: Lung nodules, Cancer, Fibrosis, Pulmonary embolism, and Severe infection.",
    "Abdominal CT": "Check specifically for: Liver disease, Pancreatitis, Kidney stones, Appendicitis, and Tumors.",
    "Spine CT": "Check specifically for: Disc disease, Spinal fractures, and Compression.",
    "Mammography": "Check specifically for: Breast cancer, Calcifications, Cysts, and Dense tissue abnormalities.",
    "Angiography": "Check specifically for: Aneurysm, Artery blockage, Vascular malformation, and Coronary disease.",
    "DEXA Scan": "Check specifically for: Osteoporosis and Bone mineral loss.",
    "Unknown": "Analyze the medical image for any visible anatomical abnormalities."
}

# ============================================================================
# STATE DEFINITION
# ============================================================================
class MedicalAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    clinical_metadata: Dict[str, Any]

# ============================================================================
# MULTI-AGENT GRAPH ENGINE
# ============================================================================
class MedicalGraphEngine:
    
    def __init__(self):
        # Vision LLM for looking at the scans (Triage & Specialist)
        # Using the new Llama 4 Scout model
        self.vision_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0.0
        )
        # Text LLM for answering follow-up questions quickly (QA)
        self.text_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.3 
        )
        
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(MedicalAgentState)
        
        # Define the three distinct agents
        workflow.add_node("triage_node", self._triage_node)
        workflow.add_node("specialist_node", self._specialist_node)
        workflow.add_node("qa_node", self._qa_node)
        
        # Routing logic
        workflow.add_conditional_edges(START, self._router)
        workflow.add_edge("triage_node", "specialist_node")
        workflow.add_edge("specialist_node", "qa_node")
        workflow.add_edge("qa_node", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _router(self, state: MedicalAgentState) -> str:
        # If an analysis is already complete, route directly to QA for follow-ups
        if state["clinical_metadata"].get("analysis_complete", False):
            return "qa_node"
        return "triage_node"

    def _triage_node(self, state: MedicalAgentState) -> dict:
        """Looks at the image and identifies the modality."""
        triage_prompt = (
            "You are a medical triage AI. Classify this image into EXACTLY ONE of these categories: "
            "[Chest X-Ray, Bone X-Ray, Abdominal X-Ray, Dental X-Ray, Brain CT, Chest CT, Abdominal CT, "
            "Spine CT, Mammography, Angiography, DEXA Scan, Unknown]. Respond ONLY with the category name."
        )
        
        last_message = state["messages"][-1]
        response = self.vision_llm.invoke([
            {"role": "system", "content": triage_prompt},
            last_message
        ])
        
        modality = response.content.strip()
        metadata = state.get("clinical_metadata", {})
        metadata["modality"] = modality
        
        return {"clinical_metadata": metadata}

    def _specialist_node(self, state: MedicalAgentState) -> dict:
        """Analyzes the image using the specific checklist for the detected modality."""
        modality = state["clinical_metadata"].get("modality", "Unknown")
        focus_areas = DISEASE_CHECKLISTS.get(modality, DISEASE_CHECKLISTS["Unknown"])
        
        system_prompt = (
            f"You are an expert radiologist specializing in {modality}. "
            f"{focus_areas}\n"
            "Provide a structured clinical report detailing your findings based ONLY on this checklist. "
            "State clearly if a condition appears to be present or absent. "
            "ALWAYS remind the user that this is an AI educational tool and NOT a substitute for professional medical advice."
        )
        
        # Find the original image message to analyze
        image_message = next((msg for msg in state["messages"] if isinstance(msg.content, list)), state["messages"][0])
        
        response = self.vision_llm.invoke([
            {"role": "system", "content": system_prompt},
            image_message
        ])
        
        # Save the report to metadata so QA node can reference it
        metadata = state["clinical_metadata"]
        metadata["diagnostic_summary"] = response.content
        metadata["analysis_complete"] = True
        
        return {
            "messages": [response],
            "clinical_metadata": metadata
        }

    def _qa_node(self, state: MedicalAgentState) -> dict:
        """Answers follow-up questions using the saved memory."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return {}

        history = state["clinical_metadata"].get("diagnostic_summary", "No prior analysis found.")
        modality = state["clinical_metadata"].get("modality", "medical scan")
        
        system_prompt = (
            f"You are a helpful medical AI assistant. Answer the patient's follow up questions. "
            f"Base your answers strictly on this initial radiologist report for a {modality}:\n{history}\n"
            "Do not hallucinate features not mentioned in the report."
        )
        
        response = self.text_llm.invoke([
            {"role": "system", "content": system_prompt},
            *state["messages"]
        ])
        
        return {"messages": [response]}

    def invoke_with_memory(self, user_message: list, thread_id: str) -> str:
        """Main entry point called by Streamlit."""
        input_state = {
            "messages": [HumanMessage(content=user_message)],
            "clinical_metadata": {}
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(input_state, config=config)
        
        final_message = result["messages"][-1]
        return final_message.content if hasattr(final_message, "content") else "Analysis complete."
