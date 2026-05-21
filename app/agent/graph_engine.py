"""
LangGraph State Engine with Memory and Multi-Agent Triage
This module routes images to a specialist based on the modality detected by the Triage Agent.
"""

import os
from typing import Annotated, Literal
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
    "Cardiac CT": "Check specifically for: Coronary artery disease, Calcification, and Heart defects.",
    "Spine CT": "Check specifically for: Disc disease, Spinal fractures, and Compression.",
    "Mammography": "Check specifically for: Breast cancer, Calcifications, Cysts, and Dense tissue abnormalities.",
    "Angiography": "Check specifically for: Aneurysm, Artery blockage, Vascular malformation, and Coronary disease.",
    "DEXA Scan": "Check specifically for: Osteoporosis and Bone mineral loss.",
    "Unknown": "Analyze the medical image for any visible anatomical abnormalities."
}

# ============================================================================
# STATE DEFINITION
# ============================================================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    patient_id: str
    analysis_history: list[str]
    modality: str  # Tracks the type of scan detected

# ============================================================================
# MULTI-AGENT GRAPH ENGINE
# ============================================================================
class MedicalGraphEngine:
    
    def __init__(self):
        # Vision LLM for looking at the scans
        self.vision_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.2-90b-vision-preview",
            temperature=0.0
        )
        # Text LLM for answering follow-up questions quickly
        self.text_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.3 
        )
        
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define the three distinct agents
        workflow.add_node("triage_agent", self._triage_node)
        workflow.add_node("specialist_agent", self._specialist_node)
        workflow.add_node("qa_agent", self._qa_node)
        
        # Routing: If analysis exists, go to QA. Otherwise, start Triage.
        workflow.add_conditional_edges(START, self._router)
        
        # Triage passes the baton straight to the Specialist
        workflow.add_edge("triage_agent", "specialist_agent")
        
        workflow.add_edge("specialist_agent", END)
        workflow.add_edge("qa_agent", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _router(self, state: AgentState) -> str:
        if state.get("analysis_history"):
            return "qa_agent"
        return "triage_agent"

    def _triage_node(self, state: AgentState) -> dict:
        """Looks at the image and identifies the modality."""
        triage_prompt = (
            "You are a medical triage AI. Classify this image into EXACTLY ONE of these categories: "
            "[Chest X-Ray, Bone X-Ray, Abdominal X-Ray, Dental X-Ray, Brain CT, Chest CT, Abdominal CT, "
            "Spine CT, Mammography, Angiography, DEXA Scan, Unknown]. Respond ONLY with the category name."
        )
        
        last_message = state["messages"][-1] # Grabs the image uploaded by the user
        response = self.vision_llm.invoke([
            {"role": "system", "content": triage_prompt},
            last_message
        ])
        
        return {"modality": response.content.strip()}

    def _specialist_node(self, state: AgentState) -> dict:
        """Analyzes the image using the specific checklist for the detected modality."""
        modality = state.get("modality", "Unknown")
        focus_areas = DISEASE_CHECKLISTS.get(modality, DISEASE_CHECKLISTS["Unknown"])
        
        system_prompt = (
            f"You are an expert radiologist specializing in {modality}. "
            f"{focus_areas}\n"
            "Provide a structured clinical report detailing your findings based ONLY on this checklist. "
            "State clearly if a condition appears to be present or absent. "
            "ALWAYS remind the user that this is an AI tool and NOT a substitute for professional medical advice."
        )
        
        last_message = state["messages"][-1]
        response = self.vision_llm.invoke([
            {"role": "system", "content": system_prompt},
            last_message
        ])
        
        # Save to history so QA node can read it later
        analysis_history = state.get("analysis_history", [])
        analysis_history.append(response.content)
        
        return {
            "messages": [response],
            "analysis_history": analysis_history
        }

    def _qa_node(self, state: AgentState) -> dict:
        """Answers follow-up questions using the saved memory."""
        history = "\n".join(state.get("analysis_history", []))
        modality = state.get("modality", "medical scan")
        
        system_prompt = (
            f"You are a helpful medical AI assistant. Answer the patient's follow up questions. "
            f"Base your answers strictly on this initial radiologist report for a {modality}:\n{history}\n"
            "Do not hallucinate features not mentioned in the report."
        )
        
        messages = state["messages"]
        response = self.text_llm.invoke([
            {"role": "system", "content": system_prompt},
            *messages
        ])
        
        return {"messages": [response]}

    def invoke_with_memory(self, user_message: str, thread_id: str, patient_id: str = "default_patient") -> str:
        """Main entry point called by streamlit_app.py"""
        input_state = {
            "messages": [HumanMessage(content=user_message)],
            "patient_id": patient_id
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(input_state, config=config)
        
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            return final_message.content
        return "Analysis complete."

    def get_patient_history(self, thread_id: str) -> list[BaseMessage]:
        """Retrieves past messages for the Streamlit sidebar."""
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state_values = self.graph.get_state(config)
            if state_values and hasattr(state_values, "values"):
                return state_values.values.get("messages", [])
        except Exception:
            pass
        return []
