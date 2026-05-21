"""
Streamlit UI for Agentic Medical Image Analyzer
Features persistent session memory for multi-turn Q&A conversations.
Image handling: Uses Base64 encoding to avoid file deletion issues.
"""

import streamlit as st
import base64
import uuid

# IMPORTANT: If your graph_engine.py is in the same folder as this file, use:
# from graph_engine import MedicalGraphEngine
# If it is inside an "app/agent/" folder, use:
from app.agent.graph_engine import MedicalGraphEngine

st.set_page_config(
    page_title="Medical AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Multi-Agent Medical Image Analyzer")
st.markdown("Stateful Autonomous AI Agent with specialized Triage and Diagnostic routing.")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "graph_engine" not in st.session_state:
    st.session_state.graph_engine = MedicalGraphEngine()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📋 Session Info")
    st.info(f"**Thread ID:** `{st.session_state.thread_id[:8]}`\n\nThis thread persists your conversation with the AI across page refreshes.")
    
    if st.button("🔄 Start New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.analysis_complete = False
        st.rerun()

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Image Upload")
    uploaded_file = st.file_uploader("Upload a medical scan (MRI, X-ray, CT)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file and not st.session_state.analysis_complete:
        st.image(uploaded_file, caption="Target Medical Scan", use_container_width=True)
        
        if st.button("Run Clinical Analysis 🤖", use_container_width=True):
            with st.spinner("Triage Agent detecting modality... Specialist Agent analyzing..."):
                
                # 1. Convert Image to Base64
                bytes_data = uploaded_file.getvalue()
                base64_image = base64.b64encode(bytes_data).decode('utf-8')
                mime_type = "image/png" if uploaded_file.name.lower().endswith("png") else "image/jpeg"
                
                # 2. Format the message exactly as Vision models require
                message_content = [
                    {"type": "text", "text": "Please analyze this medical scan."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]
                
                # 3. Invoke Graph
                try:
                    response = st.session_state.graph_engine.invoke_with_memory(
                        user_message=message_content,
                        thread_id=st.session_state.thread_id
                    )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.session_state.analysis_complete = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

with col2:
    st.subheader("💬 Interactive QA")
    st.markdown("Ask follow-up questions about the findings.")
    
    with st.container(height=500, border=True):
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
    if st.session_state.analysis_complete:
        if user_query := st.chat_input("e.g., 'What do these findings suggest?'"):
            # Display user message instantly
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.spinner("Agent is thinking..."):
                # Pass just the text query to the memory engine
                response = st.session_state.graph_engine.invoke_with_memory(
                    user_message=[{"type": "text", "text": user_query}],
                    thread_id=st.session_state.thread_id
                )
                
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.caption("📌 Upload an image and run analysis to begin the chat.")
