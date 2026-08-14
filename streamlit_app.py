import streamlit as st
import os
import uuid
import io
import base64
from PIL import Image

# -----------------------------------------------------------------------------
# BRIDGE STREAMLIT SECRETS TO OS ENVIRONMENT
# -----------------------------------------------------------------------------
if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from app.agent.graph_engine import MedicalGraphEngine
from app.cv.feature_extractor import cv_extractor

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
def get_graph_engine():
    return MedicalGraphEngine()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "scan_count" not in st.session_state:
    st.session_state.scan_count = 0

if "latest_analysis" not in st.session_state:
    st.session_state.latest_analysis = None

if "engine" not in st.session_state:
    st.session_state.engine = get_graph_engine()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def encode_image_to_base64(image: Image.Image, max_dim: int = 768) -> str:
    """Resizes and compresses image to safely fit within Groq vision API limits."""
    img = image.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.scan_count = 0
    st.session_state.latest_analysis = None
    st.rerun()

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
st.title("🏥 Multi-Agent Medical Image Analyzer")

tab1, tab2 = st.tabs(["🩺 Diagnostics & Chat", "📁 Patient Dossier"])

# =============================================================================
# TAB 1: DIAGNOSTICS & INTERACTIVE CHAT
# =============================================================================
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    # --- LEFT COLUMN: SCAN UPLOAD & ANALYSIS ---
    with col_left:
        st.subheader("📤 Scan Upload")
        uploaded_file = st.file_uploader(
            "Upload a medical scan (X-ray, CT, MRI, Dental)",
            type=["jpg", "jpeg", "png"],
            help="Supports JPG and PNG up to 200MB."
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Scan", use_container_width=True)

            if st.button("🔍 Run Autonomous Analysis", type="primary", use_container_width=True):
                with st.spinner("Executing BiomedCLIP Feature Extraction & Routing..."):
                    # 1. Run zero-shot BiomedCLIP analysis
                    img_bytes = uploaded_file.getvalue()
                    cv_results = cv_extractor.analyze_image(img_bytes)

                    # 2. Package optimized base64 payload
                    base64_img = encode_image_to_base64(image)
                    image_payload = [
                        {
                            "type": "text",
                            "text": "Analyze this medical imaging scan and provide a full clinical report."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}"
                            }
                        }
                    ]

                    # 3. Invoke LangGraph
                    extra_meta = {
                        "biomed_findings": str(cv_results.get("primary_finding", {})),
                        "detected_modality": cv_results.get("detected_modality", "Unknown")
                    }

                    report = st.session_state.engine.invoke_with_memory(
                        user_message=image_payload,
                        thread_id=st.session_state.thread_id,
                        extra_metadata=extra_meta
                    )

                    # 4. Save session state
                    st.session_state.latest_analysis = report
                    st.session_state.scan_count += 1
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()

            # Display report accordion
            if st.session_state.latest_analysis:
                st.divider()
                st.success("✅ Analysis Complete")
                with st.expander("📊 View Diagnostic Report", expanded=True):
                    st.markdown(st.session_state.latest_analysis)

    # --- RIGHT COLUMN: INTERACTIVE SPECIALIST Q&A ---
    with col_right:
        st.subheader("💬 Interactive Specialist Q&A")
        
        if not st.session_state.messages:
            st.info("Upload a scan and run the analysis to begin the clinical discussion.")
        else:
            chat_container = st.container(height=480)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if user_query := st.chat_input("Ask a follow-up question about this scan..."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                
                with st.spinner("Specialist reviewing context..."):
                    response = st.session_state.engine.invoke_with_memory(
                        user_message=user_query,
                        thread_id=st.session_state.thread_id
                    )
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.rerun()

# =============================================================================
# TAB 2: PATIENT DOSSIER & SESSION STATE
# =============================================================================
with tab2:
    st.subheader("📁 Patient Historical Records")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Thread ID", st.session_state.thread_id)
    m2.metric("Total Scans Analyzed", st.session_state.scan_count)
    m3.metric("Session Status", "Active Session" if st.session_state.scan_count > 0 else "Waiting for Scan")
    
    st.divider()
    
    if st.session_state.latest_analysis:
        st.markdown("### Active Clinical Summary")
        st.info(st.session_state.latest_analysis)
    else:
        st.caption("No historical records found for this active thread session.")
        
    st.write("")
    if st.button("🔴 End Session & Clear Memory", type="secondary"):
        reset_session()
