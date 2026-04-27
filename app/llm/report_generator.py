import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load the secret key from your .env file
load_dotenv()

class MedicalReportGenerator:
    def __init__(self):
        # We use LLaMA 3 via Groq for fast, free text generation
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant" 
        )
    
    def generate_report(self, cv_analysis):
        # 1. Get the data from the CV module
        feature = cv_analysis["detected_feature"]
        confidence = cv_analysis["confidence_percent"]
        
        # 2. Create a "Prompt Template" - this tells the AI its personality and rules
        prompt_template = PromptTemplate.from_template(
            "You are a helpful AI medical assistant.\n"
            "A computer vision model analyzed an image and detected: '{feature}' with {confidence}% confidence.\n"
            "Write a 3-sentence professional, plain-English report about this finding. Explain what this item generally is.\n"
            "Always end by reminding the user to consult a real doctor for medical advice."
        )
        
        # 3. Connect the prompt to the AI model
        chain = prompt_template | self.llm
        
        # 4. Ask the AI to write the report!
        response = chain.invoke({"feature": feature, "confidence": confidence})
        return response.content