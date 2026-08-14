"""
Phase 4: DICOM Parser & Hounsfield Unit (HU) Radiodensity Windowing Engine.
Extracts clinical PACS metadata and maps 16-bit radiodensity data to standard viewing windows.
"""

import io
import pydicom
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional

# Standard Clinical Window Presets (Window Width W, Window Center/Level L)
WINDOW_PRESETS = {
    "Auto / Default DICOM": None,
    "Bone Window (W:2000, L:500)": (2000, 500),
    "Lung Window (W:1500, L:-600)": (1500, -600),
    "Brain / Subdural Window (W:80, L:40)": (80, 40),
    "Soft Tissue / Abdomen (W:400, L:50)": (400, 50),
    "Full Dynamic Range (Linear Min-Max)": "min_max",
}


def apply_windowing(
    hu_array: np.ndarray,
    window_width: float,
    window_center: float
) -> np.ndarray:
    """Clamps Hounsfield Units into [0, 255] viewing range given Window Center (L) and Width (W)."""
    hu_min = window_center - (window_width / 2.0)
    hu_max = window_center + (window_width / 2.0)
    
    clamped = np.clip(hu_array, hu_min, hu_max)
    normalized = ((clamped - hu_min) / (hu_max - hu_min + 1e-8)) * 255.0
    return normalized.astype(np.uint8)


def process_dicom_stream(
    file_bytes: bytes,
    window_preset: str = "Auto / Default DICOM"
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Parses a raw DICOM byte buffer, extracts PACS metadata headers,
    transforms raw sensor pixels into calibrated Hounsfield Units, and applies HU windowing.
    """
    dcm = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
    
    # 1. Extract PACS Clinical Metadata
    metadata = {
        "is_dicom": True,
        "patient_age": getattr(dcm, "PatientAge", "N/A"),
        "patient_sex": getattr(dcm, "PatientSex", "N/A"),
        "modality": getattr(dcm, "Modality", "Unknown"),
        "body_part": getattr(dcm, "BodyPartExamined", "Unknown"),
        "study_description": getattr(dcm, "StudyDescription", "N/A"),
        "kvp": getattr(dcm, "KVP", "N/A"),
        "exposure_time": getattr(dcm, "ExposureTime", "N/A"),
        "photometric_interpretation": getattr(dcm, "PhotometricInterpretation", "MONOCHROME2"),
    }

    # 2. Extract Raw Pixel Array
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Convert to Hounsfield Units (HU) if Rescale parameters exist
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    hu_array = (pixel_array * slope) + intercept

    # Handle Photometric Interpretation Inversion (MONOCHROME1: 0 is White)
    if metadata["photometric_interpretation"] == "MONOCHROME1":
        hu_array = np.max(hu_array) - hu_array

    # 3. Apply HU Radiodensity Windowing
    preset_val = WINDOW_PRESETS.get(window_preset)

    if preset_val == "min_max":
        # Linear min-max scaling across entire dynamic range
        hu_min, hu_max = hu_array.min(), hu_array.max()
        img_8bit = (((hu_array - hu_min) / (hu_max - hu_min + 1e-8)) * 255.0).astype(np.uint8)
    elif isinstance(preset_val, tuple):
        # Explicit preset applied (Width, Center)
        w, c = preset_val
        img_8bit = apply_windowing(hu_array, window_width=w, window_center=c)
    else:
        # Fallback to embedded DICOM WindowCenter / WindowWidth if present
        win_center = getattr(dcm, "WindowCenter", None)
        win_width = getattr(dcm, "WindowWidth", None)
        
        # Handle multiple window values in list/DS format
        if isinstance(win_center, (list, pydicom.multival.MultiValue)):
            win_center = win_center[0]
        if isinstance(win_width, (list, pydicom.multival.MultiValue)):
            win_width = win_width[0]

        if win_center is not None and win_width is not None:
            img_8bit = apply_windowing(hu_array, float(win_width), float(win_center))
        else:
            # Safe fallback to linear normalization
            hu_min, hu_max = hu_array.min(), hu_array.max()
            img_8bit = (((hu_array - hu_min) / (hu_max - hu_min + 1e-8)) * 255.0).astype(np.uint8)

    # Convert 2D Grayscale Array to 3-Channel RGB PIL Image
    pil_image = Image.fromarray(img_8bit).convert("RGB")
    return pil_image, metadata
