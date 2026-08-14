"""
LangGraph Multi-Agent Triage Engine with Memory & Active Vision Models.
Routes images from a Triage Agent to a Specialist Agent using Dynamic Chain-of-Sight.
Includes Payload Cleaning and Reducers to prevent State Amnesia.
"""

import os
from typing import Annotated, Literal, Dict, Any, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

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
# STRUCTURED SCHEMAS
# ============================================================================
class TriageClassification(BaseModel):
    modality: Literal[
        "Chest X-Ray", "Bone X-Ray", "Abdominal X-Ray", "Dental X-Ray",
        "Brain CT", "Chest CT", "Abdominal CT", "Spine CT",
        "Mammography", "Angiography", "DEXA Scan", "Unknown"
    ] = Field(description="Exact imaging modality detected.")

# ============================================================================
# STATE REDUCERS & DEFINITION
# ============================================================================
def update_metadata(existing: dict, new: dict) -> dict:
    """Safely merges metadata so follow-up questions don't wipe the session memory."""
    if existing is None:
        return new if new is not None else {}
    if new is None:
        return existing
    res = existing.copy()
    res.update(new)
    return res

class MedicalAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    clinical_metadata: Annotated[dict, update_metadata]

# ============================================================================
# MULTI-AGENT GRAPH ENGINE
# ============================================================================
class MedicalGraphEngine:
    
    def __init__(self, api_key: str = None):
        groq_key = api_key or os.environ.get("GROQ_API_KEY")
        
        # Multimodal Vision Model on Groq
        self.vision_llm = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.2-11b-vision-preview",
            temperature=0.0
        )
        
        # Clinical Reasoning & Follow-up Q&A LLM
        self.text_llm = ChatGroq(
            api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
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
        if state.get("clinical_metadata", {}).get("analysis_complete", False):
            return "qa_node"
        return "triage_node"

    def _triage_node(self, state: MedicalAgentState) -> dict:
        triage_structured_llm = self.vision_llm.with_structured_output(TriageClassification)
        
        system_prompt = (
            "You are an expert medical triage AI. Analyze the uploaded image and "
            "categorize it strictly into one of the designated medical imaging modalities."
        )
        
        last_message = state["messages"][-1]
        try:
            structured_result: TriageClassification = triage_structured_llm.invoke([
                SystemMessage(content=system_prompt),
                last_message
            ])
            modality = structured_result.modality
        except Exception:
            modality = "Unknown"
            
        return {"clinical_metadata": {"modality": modality}}

    def _specialist_node(self, state: MedicalAgentState) -> dict:
        modality = state.get("clinical_metadata", {}).get("modality", "Unknown")
        instructions = MODALITY_INSTRUCTIONS.get(modality, MODALITY_INSTRUCTIONS["Unknown"])
        
        biomed_findings = state.get("clinical_metadata", {}).get("biomed_findings", "")
        biomed_context = f"\nBiomedCLIP Grounded Features: {biomed_findings}\n" if biomed_findings else ""
        
        system_prompt = (
            f"You are a highly vigilant, expert radiologist specializing in {modality}.\n"
            f"CRITICAL INSTRUCTION: Do NOT assume this image is normal. You must actively search for pathology using this specific radiological method:\n"
            f"1. Technique: {instructions['technique']}\n"
            f"2. Check specifically for: {instructions['checklist']}\n"
            f"{biomed_context}\n"
            f"Provide a clear, structured clinical report detailing:\n"
            f"- Modality & Visual Technique\n"
            f"- Primary Observations & Lesion Localization\n"
            f"- Differential Diagnoses\n"
            f"- Urgency (Routine / Expedited / Immediate Emergency)\n\n"
            f"ALWAYS remind the user that this is an AI clinical-decision support tool and NOT a substitute for professional medical advice."
        )
        
        # Extract the image message safely
        image_message = next(
            (msg for msg in state["messages"] if isinstance(getattr(msg, "content", None), list)),
            state["messages"][0]
        )

        if not isinstance(image_message, HumanMessage):
            image_message = HumanMessage(content=getattr(image_message, "content", str(image_message)))
        
        response = self.vision_llm.invoke([
            SystemMessage(content=system_prompt),
            image_message
        ])
        
        return {
            "messages": [response],
            "clinical_metadata": {
                "diagnostic_summary": response.content,
                "analysis_complete": True
            }
        }

    def _qa_node(self, state: MedicalAgentState) -> dict:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage):
            return {}

        history = state.get("clinical_metadata", {}).get("diagnostic_summary", "No prior analysis found.")
        
        system_prompt = (
            f"You are a clinical advisory chatbot having a conversation with a patient or physician.\n"
            f"Here is the official radiologist report that was already generated:\n"
            f"----------------------\n{history}\n----------------------\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. You are in CHAT MODE. DO NOT generate a new radiologist report. DO NOT output 'Findings' or 'Image Description'.\n"
            f"2. Answer user questions based strictly on the report context above.\n"
            f"3. Speak naturally and conversationally. If asked for next steps, suggest logical clinical follow-ups based purely on the report.\n"
            f"4. If the user asks for a specific format (e.g., 'in one word', 'bullet points'), obey that constraint strictly."
        )
        
        clean_messages = []
        for msg in state["messages"]:
            if isinstance(msg, AIMessage):
                clean_messages.append(msg)
            elif isinstance(msg, HumanMessage):
                if isinstance(msg.content, list):
                    text_only = next(
                        (item["text"] for item in msg.content if isinstance(item, dict) and item.get("type") == "text"),
                        "[User uploaded an image]"
                    )
                    clean_messages.append(HumanMessage(content=text_only))
                else:
                    clean_messages.append(msg)
        
        response = self.text_llm.invoke([
            SystemMessage(content=system_prompt),
            *clean_messages
        ])
        
        return {"messages": [response]}

    def invoke_with_memory(self, user_message: Any, thread_id: str, extra_metadata: dict = None) -> str:
        """Main entry point called by Streamlit."""
        input_state = {
            "messages": [HumanMessage(content=user_message if isinstance(user_message, list) else str(user_message))]
        }
        if extra_metadata:
            input_state["clinical_metadata"] = extra_metadata

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(input_state, config=config)
        
        final_message = result["messages"][-1]
        return final_message.content if hasattr(final_message, "content") else "Analysis complete."
