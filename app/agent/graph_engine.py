"""
LangGraph State Engine with Memory for Multi-turn Patient Q&A
This module manages stateful multi-agent reasoning with persistent memory checkpointing.
Updated for vision-capable LLM: LLaMA 3.2 90B Vision Preview
"""

import os
from typing import Annotated, Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()


class AgentState(TypedDict):
    """
    Defines the state passed through the LangGraph workflow.
    Messages are accumulated and automatically appended via add_messages reducer.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    patient_id: str
    analysis_history: list[str]


class MedicalGraphEngine:
    """
    Stateful LangGraph-powered medical analysis engine with persistent memory.
    Supports multi-turn conversations about previously analyzed images.
    Uses LLaMA 3.2 90B Vision Preview for native image understanding.
    """
    
    def __init__(self):
        """Initialize the vision-capable LLM and build the state graph with memory checkpointing."""
        
        # 1. Initialize the Vision-Capable LLM (LLaMA 3.2 90B Vision Preview via Groq)
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.2-90b-vision-preview",  # ← Vision-enabled model!
            temperature=0.3  # Lower temp for more consistent medical analysis
        )
        
        # 2. Initialize memory checkpointer (persists state between invocations)
        self.memory = MemorySaver()
        
        # 3. Build the state graph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Construct the LangGraph StateGraph for multi-turn medical reasoning."""
        
        # Create the state graph with AgentState as the message store
        workflow = StateGraph(AgentState)
        
        # 4. Define the main agent node
        workflow.add_node("medical_agent", self._agent_node)
        
        # 5. Set up routing: For vision-capable LLM, we don't need tool binding
        # The model can directly analyze images from Base64 URLs
        workflow.add_edge(START, "medical_agent")
        workflow.add_edge("medical_agent", END)
        
        # 6. Compile with memory checkpointer to enable state persistence
        graph = workflow.compile(checkpointer=self.memory)
        return graph
    
    def _agent_node(self, state: AgentState) -> dict:
        """
        The main agent reasoning node.
        Uses vision-capable LLM to directly analyze images and answer questions.
        """
        
        # Build system prompt for medical expertise
        system_prompt = (
            "You are an expert medical AI assistant specializing in medical image analysis. "
            "Your role is to:\n"
            "1. Analyze medical images that are provided in Base64 format (data:image/...;base64,...).\n"
            "2. Provide professional medical insights and explanations based on visual analysis.\n"
            "3. Reference previous analyses in the conversation to build a coherent patient history.\n"
            "4. Answer follow-up questions about the analyzed images with full context awareness.\n"
            "5. ALWAYS remind the user that this is an AI tool and NOT a substitute for professional medical advice.\n\n"
            "When analyzing an image:\n"
            "- Identify the type of medical scan\n"
            "- Describe key findings and observations\n"
            "- Note any abnormalities or areas of concern\n"
            "- Provide clinical context when relevant\n"
            "- Suggest appropriate follow-up questions or considerations"
        )
        
        # Get the messages from state
        messages = state["messages"]
        
        # Invoke the vision-capable LLM directly (no tool binding needed)
        # The model can process Base64 image URLs directly in HumanMessage
        response = self.llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                *messages,
            ]
        )
        
        # Update state with the agent's response
        return {"messages": [response]}
    
    def invoke_with_memory(
        self, 
        user_message: str, 
        thread_id: str,
        patient_id: str = "default_patient"
    ) -> str:
        """
        Invoke the graph with persistent memory across multiple turns.
        
        Args:
            user_message: The user's input (can include Base64 image URL or text question)
            thread_id: Unique identifier for this conversation session (enables memory)
            patient_id: Patient identifier for multi-patient scenarios
            
        Returns:
            The agent's final response as a string
        """
        
        # Prepare the input state with the user message
        input_state = {
            "messages": [HumanMessage(content=user_message)],
            "patient_id": patient_id,
            "analysis_history": [],
        }
        
        # Configuration with thread_id enables memory checkpointing
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke the graph with memory persistence
        result = self.graph.invoke(input_state, config=config)
        
        # Extract the final response from the last message
        final_message = result["messages"][-1]
        
        # Handle various response formats from the LLM
        if hasattr(final_message, "content"):
            if isinstance(final_message.content, str):
                return final_message.content
            elif isinstance(final_message.content, list):
                # Handle complex content blocks
                text_content = [
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in final_message.content 
                    if isinstance(block, (dict, str))
                ]
                return " ".join(text_content) if text_content else str(final_message.content)
        
        return "Analysis complete. Please review the medical report above."
    
    def get_patient_history(self, thread_id: str) -> list[BaseMessage]:
        """
        Retrieve the complete conversation history for a patient/thread.
        Useful for displaying the chat history to the user.
        """
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Access the graph's state through the checkpoint
        try:
            # Get the state values from memory
            state_values = self.graph.get_state(config)
            if state_values and hasattr(state_values, "values"):
                return state_values.values.get("messages", [])
        except Exception as e:
            print(f"Warning: Could not retrieve history: {e}")
        
        return []
