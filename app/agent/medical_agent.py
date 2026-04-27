import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver  # <-- NEW: The Memory Module!
from app.agent.tools import analyze_medical_image

load_dotenv()

class AgenticWorkflow:
    def __init__(self):
        # 1. UPGRADE INTELLIGENCE: Swap to the massive 70B model
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile" 
        )
        self.tools = [analyze_medical_image]
        
        # 2. UPGRADE MEMORY: Create a local memory filing cabinet
        self.memory = MemorySaver()
        
        # 3. Connect the memory to the agent
        self.agent = create_react_agent(
            self.llm, 
            self.tools, 
            checkpointer=self.memory # <-- NEW: Agent now has a hippocampus!
        )

    def run(self, file_path: str, thread_id: str = "patient_001"):
        # We now pass a 'thread_id' so the AI knows WHOSE memory to look at!
        prompt = (
            f"A user uploaded a new medical image saved at this path: '{file_path}'.\n"
            "1. Use your tool to analyze what is in the image.\n"
            "2. Read the results from your tool.\n"
            "3. Write a professional medical report.\n"
            "4. IMPORTANT: Check your memory. If the user uploaded images previously, "
            "reference their history to see how their overall health profile is developing."
        )
        
        print(f"--> [AGENT THOUGHT]: Accessing memory for file {thread_id}...")
        
        # 4. We pass the config so LangGraph knows which drawer to open in the filing cabinet
        config = {"configurable": {"thread_id": thread_id}}
        
        result = self.agent.invoke(
            {"messages": [("user", prompt)]},
            config=config  # <-- NEW: Triggering the memory recall
        )
        
        return result["messages"][-1].content