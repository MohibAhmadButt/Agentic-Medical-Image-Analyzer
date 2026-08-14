import streamlit as st
import os
import uuid
from PIL import Image

# Bridge Streamlit Secrets
if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.agent.graph_engine import MedicalGraphEngine
from app.cv.feature_extractor import cv_extractor

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

if "engine" not in st.session_state:
    st.session_state.engine = get_engine()


def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.scan_count = 0
    st.session_state.current_report = None
    st.session_state.cv_telemetry = None
    st.rerun()


st.title("🏥 Multi-Agent Medical Image Analyzer")

tab_diag, tab_dossier = st.tabs(["🩺 Diagnostics & Chat", "📁 Patient Dossier"])

# -----------------------------------------------------------------------------
# TAB 1: DIAGNOSTICS & CHAT
# -----------------------------------------------------------------------------
with tab_diag:
    col_left, col_right = st.columns([1, 1], gap="large")

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
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Active Scan", use_container_width=True)

            if st.button("🔍 Execute Agentic Analysis", type="primary", use_container_width=True):
                with st.spinner("Extracting zero-shot BiomedCLIP features & synthesizing report..."):
                    # In-memory feature extraction
                    img_bytes = uploaded_file.getvalue()
                    telemetry = cv_extractor.extract_features(img_bytes)
                    
                    if modality_override != "Auto-Detect (BiomedCLIP)":
                        telemetry["standard_modality"] = modality_override

                    st.session_state.cv_telemetry = telemetry

                    # LangGraph multi-agent execution
                    report = st.session_state.engine.invoke_with_memory(
                        user_query=f"Perform radiological evaluation on {telemetry['standard_modality']}.",
                        thread_id=st.session_state.thread_id,
                        extra_meta={"telemetry": telemetry}
                    )

                    st.session_state.current_report = report
                    st.session_state.scan_count += 1
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()

            if st.session_state.current_report:
                st.divider()
                st.success("✅ Diagnostic Consultation Ready")
                
                if st.session_state.cv_telemetry:
                    top_f = st.session_state.cv_telemetry["primary_finding"]
                    st.caption(f"🔬 **Vision Detection:** `{top_f['finding']}` ({top_f['confidence']}% confidence)")

                with st.expander("📄 View Official Specialist Report", expanded=True):
                    st.markdown(st.session_state.current_report)

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

# -----------------------------------------------------------------------------
# TAB 2: DOSSIER & SESSION STATE
# -----------------------------------------------------------------------------
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
