"""
LangGraph Multi-Agent Triage Engine with Memory
Routes images from a Triage Agent to a Specialist Agent using Dynamic Chain-of-Sight.
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
# SPECIALIST KNOWLEDGE BASE & RADIOLOGICAL TECHNIQUES
# ============================================================================
MODALITY_INSTRUCTIONS = {
    "Bone X-Ray": {
        "technique": "Trace the cortical outlines of every visible bone from end to end. Look specifically for discontinuities, sharp steps, angulations, or displaced fragments.",
        "checklist": "Fractures, Osteoporosis signs, Bone tumors, Arthritis (joint space narrowing), and Osteomyelitis."
    },
    "Chest X-Ray": {
        "technique": "Use the ABCDE approach: Airway (trachea midline), Breathing (lung fields clear, inspect pleural margins for air/fluid), Circulation (heart size and borders), Disability (rib fractures), Everything else (diaphragm contours).",
        "checklist": "Pneumonia, Tuberculosis, Lung cancer, COVID-19 pneumonia, Pleural effusion, Pneumothorax, COPD, Cardiomegaly, and Pulmonary edema."
    },
    "Brain CT": {
        "technique": "Evaluate for symmetry. Check for hyperdense areas (acute bleeding), hypodense areas (ischemia/edema), midline shift, mass effect, and ventricular effacement.",
        "checklist": "Stroke signs, Brain bleed (hemorrhage), Tumors, Trauma/skull fractures, and Hydrocephalus."
    },
    "Abdominal X-Ray": {
        "technique": "Assess bowel gas patterns (look for dilated loops or air-fluid levels), solid organ outlines, and look for abnormal calcifications or free air under the diaphragm.",
        "checklist": "Bowel obstruction, Kidney stones, Perforation (pneumoperitoneum), and Constipation."
    },
    "Dental X-Ray": {
        "technique": "Examine the enamel cap, dentin, and pulp chamber of each tooth. Assess the alveolar bone levels and the periodontal ligament space.",
        "checklist": "Cavities (radiolucencies), Impacted teeth, Jaw infection/abscesses, and Bone loss."
    },
    "Chest CT": {
        "technique": "Evaluate lung parenchyma using lung windows, check mediastinal structures using soft tissue windows, and assess for pulmonary emboli or masses.",
        "checklist": "Lung nodules, Cancer, Fibrosis, Pulmonary embolism, and Severe infection."
    },
    "Abdominal CT": {
        "technique": "Perform a systematic review of solid organs (liver, spleen, kidneys, pancreas), hollow viscus, vessels, and look for free fluid or lymphadenopathy.",
        "checklist": "Liver disease, Pancreatitis, Kidney stones, Appendicitis, and Tumors."
    },
    "Spine CT": {
        "technique": "Evaluate vertebral body alignment, disk spaces, facet joints, and the spinal canal for narrowing or impingement.",
        "checklist": "Disc disease, Spinal fractures, and Compression."
    },
    "Mammography": {
        "technique": "Compare bilateral symmetry. Search for spiculated masses, architectural distortion, and clusters of microcalcifications.",
        "checklist": "Breast cancer, Calcifications, Cysts, and Dense tissue abnormalities."
    },
    "Unknown": {
        "technique": "Perform a systematic visual sweep of the entire image from outside to inside, noting any asymmetries, abnormal densities, or structural disruptions.",
        "checklist": "Any visible anatomical abnormalities or signs of trauma."
    }
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
        # Vision LLM for looking at the scans (Using the new Llama 4 Scout model)
        self.vision_llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="meta-llama/llama-4-scout-17b-16e-instruct", 
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
        workflow = StateGraph(MedicalAgentState)
        
        workflow.add_node("triage_node", self._triage_node)
        workflow.add_node("specialist_node", self._specialist_node)
        workflow.add_node("qa_node", self._qa_node)
        
        workflow.add_conditional_edges(START, self._router)
        workflow.add_edge("triage_node", "specialist_node")
        workflow.add_edge("specialist_node", "qa_node")
        workflow.add_edge("qa_node", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def _router(self, state: MedicalAgentState) -> str:
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
        """Analyzes the image using a modality-specific Chain-of-Sight technique."""
        modality = state["clinical_metadata"].get("modality", "Unknown")
        instructions = MODALITY_INSTRUCTIONS.get(modality, MODALITY_INSTRUCTIONS["Unknown"])
        
        system_prompt = (
            f"You are a highly vigilant, expert radiologist specializing in {modality}. "
            f"CRITICAL INSTRUCTION: Do NOT assume this image is normal. You must actively search for pathology using this specific radiological method:\n"
            f"1. {instructions['technique']}\n"
            f"2. Check specifically for: {instructions['checklist']}\n\n"
            "Provide a structured clinical report detailing your findings. State clearly if a condition appears to be present or absent. "
            "If you see a fracture, mass, or anomaly, describe exactly where it is located. "
            "ALWAYS remind the user that this is an AI educational tool and NOT a substitute for professional medical advice."
        )
        
        image_message = next((msg for msg in state["messages"] if isinstance(msg.content, list)), state["messages"][0])
        
        response = self.vision_llm.invoke([
            {"role": "system", "content": system_prompt},
            image_message
        ])
        
        metadata = state["clinical_metadata"]
        metadata["diagnostic_summary"] = response.content
        metadata["analysis_complete"] = True
        
        return {
            "messages": [response],
            "clinical_metadata": metadata
        }

    def _qa_node(self, state: MedicalAgentState) -> dict:
        """Answers follow-up questions using the saved memory without generating new reports."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return {}

        history = state["clinical_metadata"].get("diagnostic_summary", "No prior analysis found.")
        modality = state["clinical_metadata"].get("modality", "medical scan")
        
        system_prompt = (
            f"You are a clinical advisory chatbot. You are having a conversational chat with a user about a {modality}. "
            f"Here is the official radiologist report that was already generated:\n"
            f"----------------------\n{history}\n----------------------\n"
            f"YOUR TASK: Answer the user's questions based ONLY on the report above. "
            f"DO NOT generate a new radiologist report. Speak conversationally. "
            f"If the user asks for next steps, suggest standard clinical follow-ups based purely on what the report found."
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
