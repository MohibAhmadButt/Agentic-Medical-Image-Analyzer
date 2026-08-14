import streamlit as st
import os
import uuid
from PIL import Image

# -----------------------------------------------------------------------------
# BRIDGE STREAMLIT SECRETS TO OS ENVIRONMENT
# -----------------------------------------------------------------------------
if "GROQ_API_KEY" in st.secrets and not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Import specialized engines
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

if "cv_data" not in st.session_state:
    st.session_state.cv_data = None

if "engine" not in st.session_state:
    st.session_state.engine = get_engine()

def reset_session():
    st.session_state.thread_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.scan_count = 0
    st.session_state.current_report = None
    st.session_state.cv_data = None
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

    # --- LEFT COLUMN: SCAN UPLOAD & ANALYSIS ---
    with col_left:
        st.subheader("📤 Scan Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload Scan (Pelvis/Hip X-ray, Chest X-ray, CT, MRI)",
            type=["jpg", "jpeg", "png"],
            help="Images are processed via Microsoft BiomedCLIP and LLaMA 3.3 70B."
        )

        modality_override = st.selectbox(
            "Modality Protocol",
            ["Auto-Detect (BiomedCLIP)", "Bone Radiograph / X-Ray", "Chest X-Ray", "Brain CT / MRI", "Dental Panorex", "Abdominal Scan"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Scan for Consultation", use_container_width=True)

            if st.button("🔍 Execute Agentic Analysis", type="primary", use_container_width=True):
                with st.spinner("BiomedCLIP extracting visual markers and LLaMA 3.3 synthesizing report..."):
                    # 1. Zero-shot feature extraction via BiomedCLIP
                    cv_results = cv_extractor.analyze_image(image)
                    st.session_state.cv_data = cv_results

                    chosen_modality = (
                        cv_results["standard_modality"] 
                        if modality_override == "Auto-Detect (BiomedCLIP)" 
                        else modality_override
                    )

                    primary = cv_results["primary_finding"]
                    diffs_str = ", ".join([f"{d['finding']} ({d['confidence']}%)" for d in cv_results["differential_findings"]])

                    # 2. Package telemetry metadata for the LangGraph agent
                    meta_payload = {
                        "standard_modality": chosen_modality,
                        "modality_confidence": cv_results["modality_confidence"],
                        "primary_finding": f"{primary['finding']} ({primary['confidence']}% confidence)",
                        "differentials": diffs_str
                    }

                    # 3. Invoke LangGraph multi-agent pipeline
                    report = st.session_state.engine.invoke_with_memory(
                        user_query=f"Perform complete diagnostic analysis for {chosen_modality}.",
                        thread_id=st.session_state.thread_id,
                        extra_meta=meta_payload
                    )

                    # 4. Update session state
                    st.session_state.current_report = report
                    st.session_state.scan_count += 1
                    st.session_state.messages.append({"role": "assistant", "content": report})
                    st.rerun()

            # Display findings accordion
            if st.session_state.current_report:
                st.divider()
                st.success("✅ Diagnostic Report Generated")
                
                # Show extracted BiomedCLIP metrics
                if st.session_state.cv_data:
                    top_f = st.session_state.cv_data["primary_finding"]
                    st.caption(f"🔬 **Vision Backbone Detection:** `{top_f['finding']}` ({top_f['confidence']}% confidence)")
                
                with st.expander("📄 View Official Specialist Report", expanded=True):
                    st.markdown(st.session_state.current_report)

    # --- RIGHT COLUMN: INTERACTIVE SPECIALIST Q&A ---
    with col_right:
        st.subheader("💬 Interactive Specialist Q&A")
        
        if not st.session_state.messages:
            st.info("Upload a scan and run the analysis to begin the clinical discussion.")
        else:
            chat_box = st.container(height=480)
            with chat_box:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            if prompt := st.chat_input("Ask a follow-up question (e.g. 'What are the treatment options?')"):
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.spinner("Specialist reviewing session context..."):
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
    st.subheader("📁 Patient Historical Records")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Thread ID", st.session_state.thread_id)
    m2.metric("Total Scans Analyzed", st.session_state.scan_count)
    m3.metric("Session Status", "Active Consultation" if st.session_state.scan_count > 0 else "Waiting for Scan")
    
    st.divider()
    
    if st.session_state.current_report:
        st.markdown("### Active Clinical Summary")
        st.info(st.session_state.current_report)
    else:
        st.caption("No historical records found for this active thread session.")
        
    st.write("")
    if st.button("🔴 End Session & Clear Memory", type="secondary"):
        reset_session()
