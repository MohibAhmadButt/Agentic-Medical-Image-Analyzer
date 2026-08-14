import base64
import io
import uuid
import streamlit as st
from PIL import Image

# Import the upgraded backend engines
from app.agent.graph_engine import MedicalGraphEngine
from app.cv.feature_extractor import cv_extractor

# -----------------------------------------------------------------------------
# PAGE SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Medical Image Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State
if "thread_id" not in st.session_state:
  st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
  st.session_state.messages = []

if "scan_count" not in st.session_state:
  st.session_state.scan_count = 0

if "latest_analysis" not in st.session_state:
  st.session_state.latest_analysis = None

if "engine" not in st.session_state:
  st.session_state.engine = MedicalGraphEngine()


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def encode_image_to_base64(image: Image.Image) -> str:
  buffered = io.BytesIO()
  image.save(buffered, format="JPEG")
  return base64.b64encode(buffered.getvalue()).decode("utf-8")


def reset_session():
  st.session_state.thread_id = str(uuid.uuid4())[:8]
  st.session_state.messages = []
  st.session_state.scan_count = 0
  st.session_state.latest_analysis = None
  st.session_state.engine = MedicalGraphEngine()
  st.rerun()


# -----------------------------------------------------------------------------
# UI LAYOUT: TABS & HEADER
# -----------------------------------------------------------------------------
st.title("🏥 Multi-Agent Medical Image Analyzer")

tab1, tab2 = st.tabs(["🩺 Diagnostics & Chat", "📁 Patient Dossier"])

# =============================================================================
# TAB 1: DIAGNOSTICS & INTERACTIVE CHAT
# =============================================================================
with tab1:
  col_left, col_right = st.columns([1, 1], gap="large")

  # --- LEFT COLUMN: SCAN UPLOAD & CV ANALYSIS ---
  with col_left:
    st.subheader("📤 Scan Upload")
    uploaded_file = st.file_uploader(
        "Upload a medical scan (DICOM/X-ray/MRI/CT)",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG and PNG up to 200MB.",
    )

    if uploaded_file is not None:
      image = Image.open(uploaded_file).convert("RGB")
      st.image(image, caption="Uploaded Scan", use_container_width=True)

      if st.button(
          "🔍 Run Autonomous Analysis",
          type="primary",
          use_container_width=True,
      ):
        with st.spinner("Executing BiomedCLIP Feature Extraction & Routing..."):
          # 1. Run local zero-shot BiomedCLIP analysis
          img_bytes = uploaded_file.getvalue()
          cv_results = cv_extractor.analyze_image(img_bytes)

          # 2. Package image payload for Groq Vision LLM
          base64_img = encode_image_to_base64(image)
          image_payload = [
              {
                  "type": "text",
                  "text": (
                      "Please perform a complete radiological triage and"
                      " analysis on this scan."
                  ),
              },
              {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
              },
          ]

          # 3. Invoke LangGraph with persistent memory
          extra_meta = {
              "biomed_findings": str(cv_results["primary_finding"]),
              "detected_modality": cv_results["detected_modality"],
          }

          report = st.session_state.engine.invoke_with_memory(
              user_message=image_payload,
              thread_id=st.session_state.thread_id,
              extra_metadata=extra_meta,
          )

          # 4. Save state
          st.session_state.latest_analysis = report
          st.session_state.scan_count += 1
          st.session_state.messages.append({"role": "assistant", "content": report})
          st.rerun()

      # Display BiomedCLIP quick metrics if analysis is complete
      if st.session_state.latest_analysis:
        st.divider()
        st.success("✅ Analysis Complete")
        with st.expander("📊 View Diagnostic Report", expanded=True):
          st.markdown(st.session_state.latest_analysis)

  # --- RIGHT COLUMN: INTERACTIVE SPECIALIST Q&A ---
  with col_right:
    st.subheader("💬 Interactive Specialist Q&A")

    if not st.session_state.messages:
      st.info(
          "Upload a scan and run the analysis to begin the clinical discussion."
      )
    else:
      # Render message history
      chat_container = st.container(height=480)
      with chat_container:
        for msg in st.session_state.messages:
          with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

      # Chat Input Box
      if user_query := st.chat_input("Ask a follow-up question about this scan..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate response using memory checkpointer
        with st.spinner("Specialist reviewing context..."):
          response = st.session_state.engine.invoke_with_memory(
              user_message=user_query, thread_id=st.session_state.thread_id
          )
          st.session_state.messages.append(
              {"role": "assistant", "content": response}
          )

        st.rerun()

# =============================================================================
# TAB 2: PATIENT DOSSIER & SESSION STATE
# =============================================================================
with tab2:
  st.subheader("📁 Patient Historical Records")

  m1, m2, m3 = st.columns(3)
  m1.metric("Thread ID", st.session_state.thread_id)
  m2.metric("Total Scans Analyzed", st.session_state.scan_count)
  m3.metric(
      "Session Status",
      "Active Session"
      if st.session_state.scan_count > 0
      else "Waiting for Scan",
  )

  st.divider()

  if st.session_state.latest_analysis:
    st.markdown("### Active Clinical Summary")
    st.info(st.session_state.latest_analysis)
  else:
    st.caption("No historical records found for this active thread session.")

  st.write("")
  if st.button("🔴 End Session & Clear Memory", type="secondary"):
    reset_session()
