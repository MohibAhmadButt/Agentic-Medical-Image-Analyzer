# 🏥 Agentic Medical Image Analyzer

An end-to-end autonomous AI system designed to analyze medical imagery (MRI, X-ray, CT) using **Vision-Language Foundation Models** and **Agentic Reasoning**. This project demonstrates how to bridge Computer Vision and LLMs using a stateful, memory-aware architecture.

---

## 🌟 Key Features

* **Autonomous Reasoning:** Leverages a **ReAct Agent** (LLaMA 3.3 70B) via LangGraph to interpret visual findings and generate professional reports.
* **Zero-Shot Medical Vision:** Powered by **OpenAI's CLIP**, capable of triaging diverse medical modalities (Brain, Chest, Skeletal, Dental) without task-specific retraining.
* **Stateful Patient Memory:** Implements `MemorySaver` to track patient history and context across multiple session uploads.
* **Production-Grade Guardrails:** Features strict input validation, size limits (5MB), and automatic data de-identification (temp file cleanup).
* **Full Observability:** Integrated with **LangSmith** for deep-trace analysis of the agent's "Chain of Thought" and tool execution.

---

## 🛠️ Technical Stack

* **AI Engine:** [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
* **LLM:** LLaMA-3.3-70b-Versatile (via [Groq](https://groq.com/))
* **Computer Vision:** CLIP (ViT-B/32) via [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Monitoring:** [LangSmith](https://smith.langchain.com/)
* **Language:** Python 3.10+

---

## 🏗️ Project Architecture

The system follows a modular "Engine & Shell" design:

1.  **Frontend (Streamlit):** Handles secure file uploads and visual feedback.
2.  **The Agent (LangGraph):** The central controller that manages the workflow logic and memory.
3.  **The Tools (CV Extractor):** A specialized tool that allows the agent to "see" by processing image bytes through a Foundation Model.
4.  **Observability Layer:** Sends real-time telemetry to LangSmith for debugging and performance monitoring.

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.10 or higher
* A Groq API Key (Free)
* A LangSmith API Key (Free)

### 2. Installation

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/agentic-medical-analyzer.git](https://github.com/YOUR_USERNAME/agentic-medical-analyzer.git)
cd agentic-medical-analyzer

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
### 3. Environment Setup
* Create a .env file in the root directory:
* GROQ_API_KEY=your_groq_key_here
* LANGCHAIN_TRACING_V2=true
* LANGCHAIN_API_KEY=your_langsmith_key_here
* LANGCHAIN_PROJECT="Medical-AI-Agent"

### 4. Run the Application
```bash
streamlit run streamlit_app.py

# Install dependencies
```
* pip install -r requirements.txt
