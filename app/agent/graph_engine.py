"""
LangGraph State Engine with Memory for Multi-turn Patient Q&A
This module manages stateful multi-agent reasoning with persistent memory checkpointing.
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
from app.agent.tools import analyze_medical_image

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
    """
    
    def __init__(self):
        """Initialize the LLM and build the state graph with memory checkpointing."""
        
        # 1. Initialize the LLM (LLaMA 3.3 70B via Groq)
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
            temperature=0.3  # Lower temp for more consistent medical analysis
        )
        
        # 2. Define tools available to the agent
        self.tools = [analyze_medical_image]
        
        # 3. Initialize memory checkpointer (persists state between invocations)
        self.memory = MemorySaver()
        
        # 4. Build the state graph
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Construct the LangGraph StateGraph for multi-turn medical reasoning."""
        
        # Create the state graph with AgentState as the message store
        workflow = StateGraph(AgentState)
        
        # 5. Define the main agent node
        workflow.add_node("medical_agent", self._agent_node)
        workflow.add_node("tools_executor", self._tools_node)
        
        # 6. Set up routing: if the agent wants to use tools, go to tools_executor
        # Otherwise, end the conversation
        workflow.add_conditional_edges(
            "medical_agent",
            self._should_use_tools,
            {
                "tools": "tools_executor",
                "end": END,
            },
        )
        
        # 7. After tools are used, loop back to medical_agent for analysis
        workflow.add_edge("tools_executor", "medical_agent")
        
        # 8. Set the entry point
        workflow.add_edge(START, "medical_agent")
        
        # 9. Compile with memory checkpointer to enable state persistence
        graph = workflow.compile(checkpointer=self.memory)
        return graph
    
    def _agent_node(self, state: AgentState) -> dict:
        """
        The main agent reasoning node.
        Decides whether to use tools or respond directly to the user.
        """
        
        # Build system prompt for medical expertise
        system_prompt = (
            "You are an expert medical AI assistant specializing in medical image analysis. "
            "Your role is to:\n"
            "1. Analyze medical images using the 'analyze_medical_image' tool when the user uploads an image.\n"
            "2. Provide professional medical insights and explanations.\n"
            "3. Reference previous analyses in the conversation to build a coherent patient history.\n"
            "4. Answer follow-up questions about the analyzed images.\n"
            "5. ALWAYS remind the user that this is an AI tool and NOT a substitute for professional medical advice.\n\n"
            "When analyzing an image, use the tool first, then synthesize the results into a comprehensive report."
        )
        
        # Get the last user message and prepare for the agent
        messages = state["messages"]
        
        # Invoke the LLM with tool-use capability
        response = self.llm.bind_tools(self.tools).invoke(
            [
                {"role": "system", "content": system_prompt},
                *messages,
            ]
        )
        
        # Update state with the agent's response
        return {"messages": [response]}
    
    def _tools_node(self, state: AgentState) -> dict:
        """
        Execute tools requested by the agent.
        This node processes tool calls and returns results back to the agent.
        """
        
        messages = state["messages"]
        last_message = messages[-1]
        
        # Extract tool calls from the agent's message
        if hasattr(last_message, "tool_calls"):
            tool_calls = last_message.tool_calls
        else:
            return {"messages": []}
        
        # Execute each tool call
        tool_results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            
            # Execute the analyze_medical_image tool
            if tool_name == "analyze_medical_image":
                result = analyze_medical_image.invoke(tool_input)
                tool_results.append(
                    {
                        "type": "tool",
                        "content": result,
                        "tool_use_id": tool_call["id"],
                    }
                )
        
        # Build tool result messages and add to history
        tool_result_messages = [
            AIMessage(
                content=tool_results,
                tool_calls=tool_calls,
            ) if tool_results else last_message
        ]
        
        # Also store analysis in history for context
        analysis_history = state.get("analysis_history", [])
        for result in tool_results:
            analysis_history.append(result["content"])
        
        return {
            "messages": tool_result_messages,
            "analysis_history": analysis_history,
        }
    
    def _should_use_tools(self, state: AgentState) -> Literal["tools", "end"]:
        """
        Routing logic: determine if the agent should use tools or end the conversation.
        """
        last_message = state["messages"][-1]
        
        # Check if the agent's last response contains tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "end"
    
    def invoke_with_memory(
        self, 
        user_message: str, 
        thread_id: str,
        patient_id: str = "default_patient"
    ) -> str:
        """
        Invoke the graph with persistent memory across multiple turns.
        
        Args:
            user_message: The user's input (image path or question)
            thread_id: Unique identifier for this conversation session (enables memory)
            patient_id: Patient identifier for multi-patient scenarios
            
        Returns:
            The agent's final response as a string
        """
        
        # Prepare the input state
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
        
        if hasattr(final_message, "content"):
            if isinstance(final_message.content, str):
                return final_message.content
            elif isinstance(final_message.content, list):
                # Handle tool results format
                text_content = [
                    block for block in final_message.content 
                    if isinstance(block, str)
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
