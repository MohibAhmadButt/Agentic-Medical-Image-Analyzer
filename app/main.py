from fastapi import FastAPI, File, UploadFile
import shutil
import os
from app.agent.medical_agent import AgenticWorkflow

app = FastAPI(title="Agentic Medical Image Analyzer")
os.makedirs("uploads", exist_ok=True)

# Boot up our autonomous agent
print("Booting up Agentic Workflow...")
medical_agent = AgenticWorkflow()
print("Agent Ready! 🤖")

@app.get("/")
def read_root():
    return {"message": "Agentic Server is running!"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1. Save the file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Hand the file path to the Agent and let it do EVERYTHING else!
    final_report = medical_agent.run(file_path)
    
    # 3. Return the agent's report
    return {
        "filename": file.filename,
        "agent_report": final_report
    }