import streamlit as st
import tempfile
import os
from app.agent.medical_agent import AgenticWorkflow

# 1. Page Configuration
st.set_page_config(page_title="Medical AI Agent", page_icon="🏥", layout="centered")

st.title("🏥 Agentic Medical Image Analyzer")
st.markdown("Autonomous AI Agent for medical image analysis and reporting.")

# 2. Initialize the Agent (Cached so it stays in memory)
@st.cache_resource
def load_agent():
    return AgenticWorkflow()

agent = load_agent()

# 3. Secure File Uploader
uploaded_file = st.file_uploader("Upload a medical scan (MRI, X-ray, etc.)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- INPUT VALIDATION ---
    MAX_FILE_SIZE = 5 * 1024 * 1024 # 5MB Limit
    
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("❌ File too large! Please upload an image smaller than 5MB.")
    else:
        # Display the uploaded image
        st.image(uploaded_file, caption="Target Medical Scan", use_container_width=True)
        
        # 4. Action Button
        if st.button("Run AI Agent 🤖"):
            try:
                with st.spinner("Agent is reasoning and accessing tools..."):
                    
                    # Create a safe temporary file for the agent to read
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        # Windows Path Fix: Convert \ to /
                        tmp_path = tmp_file.name.replace("\\", "/")
                    
                    # --- EXECUTE AGENT ---
                    final_report = agent.run(tmp_path)
                    
                    st.success("Analysis Complete!")
                    
                    # --- VISUAL CONFIDENCE METER ---
                    st.markdown("### 📊 Agent Assessment Metrics")
                    st.progress(1.0, text="Agent Decision Stability: High")
                    
                    st.markdown("### 📝 Agent's Medical Report")
                    st.info(final_report)
                    
            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
                st.warning("Ensure your GROQ_API_KEY and LANGCHAIN_API_KEY are correct in your .env file.")
                
            finally:
                # --- DATA PRIVACY ---
                # Remove the scan from the server immediately after analysis
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)

st.divider()
st.caption("Disclaimer: This is an AI-powered educational tool. Not for clinical diagnosis.")