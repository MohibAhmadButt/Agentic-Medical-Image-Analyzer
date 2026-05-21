"""
Streamlit UI for Agentic Medical Image Analyzer
Features an Enterprise Tabbed UI, Quick Actions, and a structured Patient Dossier.
"""

import streamlit as st
import base64
import uuid
from app.agent.graph_engine import MedicalGraphEngine

st.set_page_config(
    page_title="Medical AI Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar for a cleaner main view
)

st.title("🏥 Multi-Agent Medical Image Analyzer")

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
if "patient_records" not in st.session_state:
    st.session_state.patient_records = [] # To store structured records

# ============================================================================
# MAIN CONTENT: TABBED INTERFACE
# ============================================================================
tab_diagnostics, tab_records = st.tabs(["🩺 Diagnostics & Chat", "🗂️ Patient Dossier"])

with tab_diagnostics:
    col_image, col_chat = st.columns([1, 1.2], gap="large")

    with col_image:
        st.subheader("📤 Scan Upload")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Medical Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            
            if uploaded_file:
                st.image(uploaded_file, use_container_width=True)
                
                if not st.session_state.analysis_complete:
                    if st.button("Start AI Triage & Analysis 🚀", use_container_width=True, type="primary"):
                        with st.spinner("Triage Agent detecting modality..."):
                            # Convert Image to Base64
                            bytes_data = uploaded_file.getvalue()
                            base64_image = base64.b64encode(bytes_data).decode('utf-8')
                            mime_type = "image/png" if uploaded_file.name.lower().endswith("png") else "image/jpeg"
                            
                            message_content = [
                                {"type": "text", "text": "Please analyze this medical scan."},
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                            ]
                            
                            try:
                                response = st.session_state.graph_engine.invoke_with_memory(
                                    user_message=message_content,
                                    thread_id=st.session_state.thread_id
                                )
                                # Save to Chat
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                                # Save to Patient Records
                                st.session_state.patient_records.append({"file": uploaded_file.name, "report": response})
                                st.session_state.analysis_complete = True
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error during analysis: {e}")

    with col_chat:
        st.subheader("💬 Interactive Specialist Q&A")
        
        # Chat History Container
        chat_container = st.container(height=500, border=True)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("Upload a scan and run the analysis to begin the clinical discussion.")
            
            for msg in st.session_state.chat_history:
                # Use custom avatars
                avatar = "🧑‍⚕️" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.write(msg["content"])
                    
        # Quick Actions & Chat Input
        if st.session_state.analysis_complete:
            # Quick Action Buttons
            col_q1, col_q2 = st.columns(2)
            quick_query = None
            if col_q1.button("Explain in simple terms", use_container_width=True):
                quick_query = "Can you explain these findings in simple, non-medical terms?"
            if col_q2.button("What are the next steps?", use_container_width=True):
                quick_query = "Based on this report, what are the recommended next steps or tests?"

            # Input handling (either from quick buttons or manual typing)
            user_query = st.chat_input("Ask a follow-up question...") or quick_query
            
            if user_query:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                with chat_container:
                    with st.chat_message("user", avatar="👤"):
                        st.write(user_query)
                
                with st.spinner("Specialist is reviewing history..."):
                    response = st.session_state.graph_engine.invoke_with_memory(
                        user_message=[{"type": "text", "text": user_query}],
                        thread_id=st.session_state.thread_id
                    )
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

with tab_records:
    st.subheader("🗂️ Patient Historical Records")
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    col_metrics1.metric("Thread ID", f"{st.session_state.thread_id[:8]}")
    col_metrics2.metric("Total Scans Analyzed", len(st.session_state.patient_records))
    col_metrics3.metric("Status", "Active Session" if st.session_state.analysis_complete else "Waiting for Scan")
    
    st.divider()
    
    if st.session_state.patient_records:
        for idx, record in enumerate(reversed(st.session_state.patient_records)):
            with st.expander(f"📄 Report {len(st.session_state.patient_records) - idx}: {record['file']}", expanded=(idx==0)):
                st.markdown(record['report'])
    else:
        st.caption("No historical records found for this session.")
        
    st.divider()
    if st.button("🛑 End Session & Clear Memory", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.patient_records = []
        st.session_state.analysis_complete = False
        st.rerun()
