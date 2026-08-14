import streamlit as st
import os
import uuid
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# SECRETS & ENVIRONMENT CONFIGURATION
# -----------------------------------------------------------------------------
if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.agent.graph_engine import MedicalGraphEngine

# -----------------------------------------------------------------------------
# PAGE SETUP & CACHED GRAPH INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Medical Image Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def get_engine():
    return MedicalGraphEngine()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

if "current_report" not in st.session_state:
    st.session_state.current_report = None

if "engine" not in st.session_state:
    st.session_state.engine = get_engine()

def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.scan_count = 0
    st.session_state.current_report = None
    st.rerun()

# -----------------------------------------------------------------------------
# LIGHTWEIGHT VISUAL TELEMETRY EXTRACTOR (ZERO API DEPENDENCY)
# -----------------------------------------------------------------------------
def extract_scan_telemetry(image: Image.Image) -> dict:
    """Computes instant visual biomarkers (density, contrast, aspect ratio) safely."""
    grayscale = image.convert("L")
    arr = np.array(grayscale)
    
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    
    # Heuristic modality detection based on aspect ratio and contrast profiles
    w, h = image.size
    aspect_ratio = round(w / float(h), 2)

    if aspect_ratio > 1.4:
        suggested_modality = "Dental Panorex"
    elif mean_val < 70 and std_val > 50:
        suggested_modality = "Brain CT / MRI"
    elif std_val < 35:
        suggested_modality = "Bone Radiograph / X-Ray"
    else:
        suggested_modality = "Chest X-Ray"

    return {
        "mean_intensity": round(mean_val, 2),
        "contrast_deviation": round(std_val, 2),
        "aspect_ratio": aspect_ratio,
        "suggested_modality": suggested_modality
    }

# -----------------------------------------------------------------------------
# UI TABS
# -----------------------------------------------------------------------------
st.title("🏥 Multi-Agent Medical Image Analyzer")

tab_diag, tab_dossier = st.tabs(["🩺 Diagnostics & Chat", "📁 Patient Dossier"])

# =============================================================================
# TAB 1: DIAGNOSTICS & CHAT
# =============================================================================
with tab_diag:
    col_left, col_right = st.columns([1, 1], gap="large")

    # --- LEFT PANE: UPLOAD & ANALYSIS ---
    with col_left:
        st.subheader("📤 Scan Upload & Triage")
        
        uploaded_file = st.file_uploader(
            "Upload Scan (DICOM Export, X-ray, CT, MRI)",
            type=["jpg", "jpeg", "png"],
            help="Images are processed with local privacy-preserving telemetry."
        )

        modality_override = st.selectbox(
            "Target Modality / Protocol Preset",
            ["Auto-Detect", "Chest X-Ray", "Brain CT / MRI", "Bone Radiograph / X-Ray", "Dental Panorex", "Abdominal Scan"]
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Active Patient Scan", use_container_width=True)

            if st.button("🔍 Execute Agentic Analysis", type="primary", use_container_width=True):
                with st.spinner("Extracting visual biomarkers & executing specialist agent reasoning..."):
                    telemetry = extract_scan_telemetry(img)
                    
                    target_modality = telemetry["suggested_modality"] if modality_override == "Auto-Detect" else modality_override
                    
                    meta_payload = {
                        "modality_hint": target_modality,
                        "findings_summary": (
                            f"Detected Modality: {target_modality}. "
                            f"Mean Optical Density: {telemetry['mean_intensity']}, "
                            f"Tissue Contrast Standard Dev: {telemetry['contrast_deviation']}, "
                            f"Aspect Ratio: {telemetry['aspect_ratio']}."
                        )
                    }

                    report = st.session_state.engine.invoke_with_memory(
                        user_query=f"Analyze {target_modality} scan.",
                        thread_id=st.session_state.thread_id,
                        extra_meta=meta_payload
                    )

                    st.session_state.current_report = report
                    st.session_state.scan_count += 1
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()

            if st.session_state.current_report:
                st.divider()
                st.success("✅ Diagnostic Consultation Ready")
                with st.expander("📄 View Official Specialist Report", expanded=True):
                    st.markdown(st.session_state.current_report)

    # --- RIGHT PANE: CONVERSATIONAL SPECIALIST ---
    with col_right:
        st.subheader("💬 Interactive Specialist Q&A")

        if not st.session_state.messages:
            st.info("Upload an imaging scan and run analysis to begin the clinical discussion.")
        else:
            chat_box = st.container(height=480)
            with chat_box:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if prompt := st.chat_input("Ask a question about this scan or diagnosis..."):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Consulting specialist memory..."):
                    response = st.session_state.engine.invoke_with_memory(
                        user_query=prompt,
                        thread_id=st.session_state.thread_id
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})

                st.rerun()

# =============================================================================
# TAB 2: DOSSIER & MEMORY INSPECTOR
# =============================================================================
with tab_dossier:
    st.subheader("📁 Patient Session Dossier")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Thread Identifier", st.session_state.thread_id)
    m2.metric("Scans Analyzed", st.session_state.scan_count)
    m3.metric("Session State", "Active Consultation" if st.session_state.scan_count > 0 else "Awaiting Input")

    st.divider()

    if st.session_state.current_report:
        st.markdown("### Active Clinical Summary")
        st.info(st.session_state.current_report)
    else:
        st.caption("No historical records found for this active thread session.")

    st.write("")
    if st.button("🔴 End Session & Purge Checkpointer Memory", type="secondary"):
        reset_session()
