import json
from typing import Union
from app.cv.feature_extractor import cv_extractor
from langchain_core.tools import tool


@tool
def run_biomed_vision_analysis(image_path: str) -> str:
  """Analyzes an uploaded medical scan using BiomedCLIP foundation vision model.

  Returns detected modality, primary findings, and differential diagnosis
  probabilities.
  """
  try:
    results = cv_extractor.analyze_image(image_path)
    return json.dumps(results, indent=2)
  except Exception as e:
    return json.dumps(
        {"error": f"Failed to execute vision feature extraction: {str(e)}"}
    )
