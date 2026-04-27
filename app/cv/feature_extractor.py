from transformers import pipeline
from PIL import Image
import io

class MedicalFeatureExtractor:
    def __init__(self):
        # 1. Load an official, bulletproof Foundation Model (OpenAI's CLIP)
        print("Downloading Foundation Model... (This will definitely work!)")
        
        # We use zero-shot classification, which lets us define our own labels on the fly!
        self.classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

    def extract(self, image_bytes):
        # 2. Open the image so Python can read it
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 3. DEFINE our own categories! (The Ultimate Triage List)
        candidate_labels = [
            # --- Brain & Head ---
            "healthy normal brain MRI",
            "brain MRI showing a tumor or lesion",
            
            # --- Chest & Lungs ---
            "chest x-ray showing normal healthy lungs",
            "chest x-ray showing pneumonia or fluid in the lungs",
            
            # --- Bones & Skeletal ---
            "x-ray of a normal healthy bone or joint",
            "x-ray showing a broken bone or fracture",
            "dental x-ray of teeth",
            
            # --- Skin (Dermatology) ---
            "photograph of normal healthy human skin",
            "dermatology photograph of a skin rash or eczema",
            "dermatology photograph of a dark skin lesion or melanoma",
            
            # --- Eyes (Ophthalmology) ---
            "retinal fundus photograph of a healthy eye",
            "retinal fundus photograph showing eye disease or diabetic retinopathy",
            
            # --- Microscopic & Labs ---
            "microscopic slide of blood cells or tissue",
            
            # --- THE SAFETY NET ---
            "a non-medical everyday photograph of a random object, animal, or scenery"
        ]
        
        # 4. Ask the AI to classify the image based ONLY on our custom labels
        results = self.classifier(image, candidate_labels=candidate_labels)
        
        # 5. Extract the highest confidence result
        # The pipeline returns a list sorted from highest to lowest confidence
        top_result = results[0]
        category_name = top_result['label'].title()
        score = top_result['score']
        
        return {
            "detected_feature": category_name, 
            "confidence_percent": round(score * 100, 2)
        }