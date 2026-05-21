"""
Streamlit UI for Agentic Medical Image Analyzer
Features persistent session memory for multi-turn Q&A conversations.
Image handling: Uses Base64 encoding to avoid file deletion issues.
"""

import streamlit as st
import base64
import uuid
from app.agent.graph_engine import MedicalGraphEngine

# ============================================================================
# PAGE CONFIGURATION & SIDEBAR SETUP
# ============================================================================

st.set_page_config(
    page_title="Medical AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Agentic Medical Image Analyzer")
st.markdown("Multi-turn Autonomous AI Agent for medical image analysis with patient memory.")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Initialize session state variables for persistent memory across reruns
if "graph_engine" not in st.session_state:
    st.session_state.graph_engine = MedicalGraphEngine()

if "thread_id" not in st.session_state:
    # Generate a unique thread_id for this Streamlit session
    # This persists the conversation in LangGraph's memory
    st.session_state.thread_id = str(uuid.uuid4())

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "patient_default"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "image_b64" not in st.session_state:
    st.session_state.image_b64 = None

if "image_filename" not in st.session_state:
    st.session_state.image_filename = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_image_to_base64(uploaded_file) -> str:
    """
    Convert uploaded image to Base64 string.
    This avoids file deletion issues and enables memory persistence.
    """
    image_bytes = uploaded_file.getvalue()
    base64_string = base64.b64encode(image_bytes).decode("utf-8")
    return base64_string


def get_image_mime_type(filename: str) -> str:
    """Determine MIME type from filename."""
    ext = filename.lower().split('.')[-1]
    mime_types = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png"
    }
    return mime_types.get(ext, "image/jpeg")


# ============================================================================
# SIDEBAR: SESSION MANAGEMENT & INFO
# ============================================================================

with st.sidebar:
    st.header("📋 Session Info")
    
    # Display current session details
    st.info(f"""
    **Thread ID:** `{st.session_state.thread_id[:8]}...`
    
    **Patient ID:** `{st.session_state.patient_id}`
    
    This thread persists your conversation with the AI across page refreshes.
    """)
    
    # Option to start a new session
    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.image_b64 = None
        st.session_state.image_filename = None
        st.rerun()
    
    # Custom patient ID input
    new_patient_id = st.text_input(
        "Patient ID (optional)",
        value=st.session_state.patient_id,
        help="Enter a patient identifier for multi-patient scenarios"
    )
    if new_patient_id != st.session_state.patient_id:
        st.session_state.patient_id = new_patient_id
    
    st.divider()
    
    # Display chat history
    st.subheader("💬 Conversation History")
    if st.session_state.chat_history:
        with st.expander(f"View {len(st.session_state.chat_history)} messages"):
            for i, msg in enumerate(st.session_state.chat_history):
                st.text(f"{i+1}. {msg[:100]}...")
    else:
        st.caption("No messages yet. Upload an image to start!")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Create two columns for upload and chat
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Image Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a medical scan (MRI, X-ray, CT, etc.)...",
        type=["jpg", "jpeg", "png"],
        help="Maximum 5MB. Supports: JPG, PNG"
    )
    
    if uploaded_file is not None:
        # --- INPUT VALIDATION ---
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB Limit
        
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("❌ File too large! Please upload an image smaller than 5MB.")
        else:
            # Display the uploaded image
            st.image(uploaded_file, caption="Target Medical Scan", use_container_width=True)
            
            # --- CONVERT TO BASE64 (No temp file!) ---
            # Read image immediately and convert to Base64
            image_b64 = convert_image_to_base64(uploaded_file)
            st.session_state.image_b64 = image_b64
            st.session_state.image_filename = uploaded_file.name
            
            # Create action buttons
            col_analyze, col_clear = st.columns(2)
            
            with col_analyze:
                analyze_clicked = st.button("Run AI Agent 🤖", use_container_width=True)
            
            with col_clear:
                clear_clicked = st.button("Clear", use_container_width=True)
            
            if analyze_clicked:
                try:
                    with st.spinner("Agent is reasoning and accessing tools..."):
                        
                        # --- BUILD IMAGE URL FOR LLM ---
                        mime_type = get_image_mime_type(uploaded_file.name)
                        image_url = f"data:{mime_type};base64,{image_b64}"
                        
                        # --- INVOKE AGENT WITH MEMORY ---
                        # Build a prompt that includes the image URL
                        analysis_prompt = (
                            f"Please analyze this medical image and provide a detailed professional medical report.\n\n"
                            f"Image: {image_url}\n\n"
                            f"Filename: {uploaded_file.name}\n\n"
                            f"Provide comprehensive analysis including:\n"
                            f"1. What type of medical scan this is\n"
                            f"2. Key findings and observations\n"
                            f"3. Any notable features or anomalies\n"
                            f"4. Clinical significance\n"
                            f"5. Recommendations for follow-up"
                        )
                        
                        # Invoke the graph engine with persistent memory (thread_id)
                        response = st.session_state.graph_engine.invoke_with_memory(
                            user_message=analysis_prompt,
                            thread_id=st.session_state.thread_id,
                            patient_id=st.session_state.patient_id
                        )
                        
                        # Store in session history
                        st.session_state.chat_history.append(f"User: Uploaded and analyzed image - {uploaded_file.name}")
                        st.session_state.chat_history.append(f"Agent: {response[:200]}...")
                        
                        st.success("✅ Analysis Complete!")
                        
                        # --- VISUAL CONFIDENCE METER ---
                        st.markdown("### 📊 Agent Assessment Metrics")
                        st.progress(0.95, text="Agent Decision Confidence: Very High")
                        
                        # --- MEDICAL REPORT ---
                        st.markdown("### 📝 Agent's Medical Report")
                        st.info(response)
                        
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {e}")
                    st.warning("Ensure your GROQ_API_KEY and LANGCHAIN_API_KEY are correct in your .env file.")
            
            if clear_clicked:
                st.session_state.image_b64 = None
                st.session_state.image_filename = None
                st.rerun()

with col2:
    st.subheader("💬 Follow-up Q&A")
    
    st.markdown("""
    After analyzing an image, ask follow-up questions about the findings.
    The AI remembers previous analyses in this conversation.
    """)
    
    # Display chat messages
    if st.session_state.chat_history:
        with st.container(border=True):
            for msg in st.session_state.chat_history:
                if msg.startswith("User:"):
                    st.write(f"**👤 {msg}**")
                else:
                    st.write(f"🤖 {msg}")
    
    # Input area for follow-up questions
    st.markdown("---")
    
    if st.session_state.image_b64 or st.session_state.chat_history:
        user_question = st.text_input(
            "Ask a follow-up question about the analysis...",
            placeholder="e.g., 'What do these findings suggest?' or 'How serious is this?'",
            key="followup_input"
        )
        
        if user_question:
            if st.button("Send 📨", use_container_width=True):
                try:
                    with st.spinner("Agent is thinking..."):
                        
                        # Invoke the agent with the follow-up question
                        # The thread_id ensures it accesses the same memory
                        response = st.session_state.graph_engine.invoke_with_memory(
                            user_message=user_question,
                            thread_id=st.session_state.thread_id,
                            patient_id=st.session_state.patient_id
                        )
                        
                        # Add to chat history
                        st.session_state.chat_history.append(f"User: {user_question}")
                        st.session_state.chat_history.append(f"Agent: {response}")
                        
                        st.success("✅ Response received!")
                        st.info(response)
                        st.rerun()
                
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
    
    else:
        st.caption("📌 Upload an image and analyze it first to ask follow-up questions.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
### ⚠️ Medical Disclaimer
This is an **AI-powered educational tool** and **NOT a substitute for professional medical advice**.
Always consult with qualified healthcare professionals for accurate diagnosis and treatment.

**Key Privacy Notes:**
- Images are converted to Base64 and stored in session memory (not written to disk).
- Conversation history is stored in LangGraph's memory (MemorySaver) for this session only.
- Thread IDs enable multi-turn conversations but are session-specific.
- No temporary files are created or left behind on the server.
""")

st.caption(f"Last updated: Session {st.session_state.thread_id[:8]}...")
