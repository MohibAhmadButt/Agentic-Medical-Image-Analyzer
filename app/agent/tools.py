from langchain_core.tools import tool
from app.cv.feature_extractor import MedicalFeatureExtractor

# Load our CV model
cv_extractor = MedicalFeatureExtractor()

# The @tool decorator tells LangChain: "This is a tool the AI can use!"
@tool
def analyze_medical_image(file_path: str) -> str:
    """Use this tool to analyze a medical image. Pass the file_path to get the analysis."""
    print(f"--> [AGENT ACTION]: Using CV Tool on {file_path}")
    
    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
        
    results = cv_extractor.extract(image_bytes)
    return f"I analyzed the image. Detected: {results['detected_feature']} with {results['confidence_percent']}% confidence."