"""
FastAPI In-Memory Streaming Entrypoint.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from app.cv.feature_extractor import cv_extractor
from app.agent.graph_engine import MedicalGraphEngine

app = FastAPI(title="Agentic Medical Image Analyzer API", version="2.0")
engine = MedicalGraphEngine()


@app.get("/")
def health_check():
    return {"status": "online", "model": "BiomedCLIP + LLaMA 3.3 70B via LangGraph"}


@app.post("/analyze")
async def analyze_scan(file: UploadFile = File(...), thread_id: str = "default-session"):
    """Streams image bytes in-memory to prevent disk I/O bottlenecks."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (PNG, JPG, JPEG).")

    try:
        image_bytes = await file.read()
        telemetry, _ = cv_extractor.extract_features(image_bytes)

        report_markdown = engine.invoke_with_memory(
            user_query=f"Analyze uploaded scan with modality: {telemetry['standard_modality']}",
            thread_id=thread_id,
            extra_meta={"telemetry": telemetry}
        )

        return {
            "filename": file.filename,
            "telemetry": telemetry,
            "clinical_report": report_markdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")
