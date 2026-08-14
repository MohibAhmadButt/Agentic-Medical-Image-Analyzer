import streamlit as st
import os
import uuid
from PIL import Image

# -----------------------------------------------------------------------------
# BRIDGE STREAMLIT SECRETS TO OS ENVIRONMENT
# -----------------------------------------------------------------------------
if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.agent.graph_engine import MedicalGraphEngine
from app.cv.feature_extractor import cv_extractor
from app.utils.pdf_generator import build_clinical_pdf

# -----------------------------------------------------------------------------
# PAGE SETUP & RESOURCE CACHING
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

if "cv_telemetry" not in st.session_state:
    st.session_state.cv_telemetry = None

if "heatmap_image" not in st.session_state:
    st.session_state.heatmap_image = None

if "engine" not in st.session_state:
    st.session_state.engine = get_engine()


def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.scan_count = 0
    st.session_state.current_report = None
    st.session_state.cv_telemetry = None
    st.session_state.heatmap_image = None
    st.rerun()


# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
st.title("🏥 Multi-Agent Medical Image Analyzer")

tab_diag, tab_dossier = st.tabs(["🩺 Diagnostics & Chat", "📁 Patient Dossier"])

# =============================================================================
# TAB 1: DIAGNOSTICS & CHAT
# =============================================================================
with tab_diag:
    col_left, col_right = st.columns([1, 1], gap="large")

    # --- LEFT COLUMN: SCAN UPLOAD & VISUAL LOCALIZATION ---
    with col_left:
        st.subheader("📤 Scan Upload & Triage")
        uploaded_file = st.file_uploader(
            "Upload Scan (Pelvis/Hip X-ray, Chest X-ray, Brain CT/MRI)",
            type=["jpg", "jpeg", "png"]
        )

        modality_override = st.selectbox(
            "Modality Protocol",
            ["Auto-Detect (BiomedCLIP)", "Bone Radiograph / X-Ray", "Chest X-Ray", "Brain CT / MRI", "Dental Panorex", "Abdominal Scan"]
        )

        if uploaded_file is not None:
            raw_img = Image.open(uploaded_file).convert("RGB")

            if st.button("🔍 Execute Agentic Analysis", type="primary", use_container_width=True):
                with st.spinner("Extracting zero-shot BiomedCLIP features & generating attention heatmap..."):
                    # 1. Feature Extraction & Visual Saliency
                    telemetry, heatmap_overlay = cv_extractor.extract_features(raw_img)
                    
                    if modality_override != "Auto-Detect (BiomedCLIP)":
                        telemetry["standard_modality"] = modality_override

                    st.session_state.cv_telemetry = telemetry
                    st.session_state.heatmap_image = heatmap_overlay

                    # 2. Multi-Agent Reasoning via LangGraph
                    report = st.session_state.engine.invoke_with_memory(
                        user_query=f"Perform radiological evaluation on {telemetry['standard_modality']}.",
                        thread_id=st.session_state.thread_id,
                        extra_meta={"telemetry": telemetry}
                    )

                    st.session_state.current_report = report
                    st.session_state.scan_count += 1
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()

            # Side-by-Side Visual Inspection (Original vs Heatmap)
            if st.session_state.heatmap_image is not None:
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(raw_img, caption="Original Scan", use_container_width=True)
                with img_col2:
                    st.image(st.session_state.heatmap_image, caption="AI Attention Heatmap Overlay", use_container_width=True)
            else:
                st.image(raw_img, caption="Active Patient Scan", use_container_width=True)

            # Diagnostic Report Display
            if st.session_state.current_report and st.session_state.cv_telemetry:
                st.divider()
                st.success("✅ Diagnostic Consultation Ready")
                
                top_f = st.session_state.cv_telemetry["primary_finding"]
                st.caption(f"🔬 **Vision Detection:** `{top_f['finding']}` ({top_f['confidence']}% confidence)")

                with st.expander("📄 View Official Specialist Report", expanded=True):
                    st.markdown(st.session_state.current_report)

                # PDF Export Button
                if st.session_state.heatmap_image is not None:
                    pdf_bytes = build_clinical_pdf(
                        thread_id=st.session_state.thread_id,
                        modality=st.session_state.cv_telemetry["standard_modality"],
                        primary_finding=top_f["finding"],
                        confidence=top_f["confidence"],
                        report_markdown=st.session_state.current_report,
                        original_img=raw_img,
                        heatmap_img=st.session_state.heatmap_image
                    )

                    st.download_button(
                        label="📥 Download Official Clinical PDF Report",
                        data=pdf_bytes,
                        file_name=f"Consultation_Report_{st.session_state.thread_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

    # --- RIGHT COLUMN: INTERACTIVE SPECIALIST Q&A ---
    with col_right:
        st.subheader("💬 Interactive Specialist Q&A")

        if not st.session_state.messages:
            st.info("Upload a scan and execute analysis to begin the clinical discussion.")
        else:
            chat_box = st.container(height=480)
            with chat_box:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if prompt := st.chat_input("Ask a question about this diagnosis or treatment options..."):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Specialist reviewing context..."):
                    response = st.session_state.engine.invoke_with_memory(
                        user_query=prompt,
                        thread_id=st.session_state.thread_id
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})

                st.rerun()

# =============================================================================
# TAB 2: DOSSIER & SESSION STATE
# =============================================================================
with tab_dossier:
    st.subheader("📁 Patient Historical Records")

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
