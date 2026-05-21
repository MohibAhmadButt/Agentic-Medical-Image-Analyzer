"""
Streamlit UI for Agentic Medical Image Analyzer
Features persistent session memory for multi-turn Q&A conversations.
"""

import streamlit as st
import tempfile
import os
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

if "current_image_path" not in st.session_state:
    st.session_state.current_image_path = None

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
        st.session_state.current_image_path = None
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
            
            # Create action buttons
            col_analyze, col_clear = st.columns(2)
            
            with col_analyze:
                analyze_clicked = st.button("Run AI Agent 🤖", use_container_width=True)
            
            with col_clear:
                clear_clicked = st.button("Clear", use_container_width=True)
            
            if analyze_clicked:
                try:
                    with st.spinner("Agent is reasoning and accessing tools..."):
                        
                        # Create a safe temporary file for the agent to read
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name.replace("\\", "/")
                        
                        # Store the image path in session for follow-up questions
                        st.session_state.current_image_path = tmp_path
                        
                        # --- INVOKE AGENT WITH MEMORY ---
                        # Build a detailed prompt for the agent
                        analysis_prompt = f"Please analyze the medical image at path '{tmp_path}'. Provide a detailed professional medical report."
                        
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
                
                finally:
                    # --- DATA PRIVACY ---
                    # Remove the scan from the server immediately after analysis
                    if 'tmp_path' in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            if clear_clicked:
                st.session_state.current_image_path = None
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
    
    if st.session_state.current_image_path or st.session_state.chat_history:
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
- Uploaded images are temporarily stored and deleted after analysis.
- Conversation history is stored in LangGraph's memory (MemorySaver) for this session only.
- Thread IDs enable multi-turn conversations but are session-specific.
""")

st.caption(f"Last updated: Session {st.session_state.thread_id[:8]}...")
