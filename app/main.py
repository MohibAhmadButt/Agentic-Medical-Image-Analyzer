"""
FastAPI In-Memory Streaming Entrypoint with DICOM & Standard Image Ingestion.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from app.cv.feature_extractor import cv_extractor
from app.agent.graph_engine import MedicalGraphEngine

app = FastAPI(title="Agentic Medical Image Analyzer API", version="2.0")
engine = MedicalGraphEngine()


@app.get("/")
def health_check():
    return {"status": "online", "model": "BiomedCLIP + LLaMA 3.3 70B via LangGraph (DICOM Supported)"}


@app.post("/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    window_preset: str = "Auto / Default DICOM",
    thread_id: str = "default-session"
):
    """Streams DICOM or standard image bytes in-memory for zero disk I/O latency."""
    filename = file.filename.lower()
    valid_extensions = (".dcm", ".png", ".jpg", ".jpeg")
    
    if not any(filename.endswith(ext) for ext in valid_extensions) and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a DICOM (.dcm) or standard image (PNG, JPG, JPEG)."
        )

    try:
        file_bytes = await file.read()
        telemetry, _, _ = cv_extractor.extract_features(file_bytes, window_preset=window_preset)

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
