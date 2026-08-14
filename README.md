# 🏥 Multi-Agent Medical Image Analyzer

### Autonomous, Explainable Multimodal Clinical Decision-Support System

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-medical-image-analyzer-rxq6j3nyuzfprxwpau4p3x.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python\&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-BiomedCLIP-EE4C2C.svg?logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-FF4B4B.svg)](https://github.com/langchain-ai/langgraph)
[![LLaMA 3.3](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-F55036.svg)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[🚀 Live Demo](https://agentic-medical-image-analyzer-rxq6j3nyuzfprxwpau4p3x.streamlit.app/)** • **[📂 GitHub Repository](https://github.com/MohibAhmadButt/Agentic-Medical-Image-Analyzer)**

---

## 📌 Overview

**Multi-Agent Medical Image Analyzer** is an end-to-end multimodal AI application for analyzing medical imagery and generating explainable, structured clinical-style reports.

The system combines **Microsoft BiomedCLIP**, multi-resolution image analysis, **Grad-CAM visual explanations**, **LangGraph multi-agent orchestration**, clinical guideline retrieval, and **Meta LLaMA 3.3 70B** reasoning.

It supports medical imaging workflows involving **X-rays, CT, MRI, Panorex, and DICOM files**, transforming raw image data into structured vision telemetry that can be processed by specialized AI agents.

> **This project is designed for research and educational purposes. It is not a medical diagnostic device and must not be used as a substitute for qualified medical professionals.**

---

## ✨ Features

* 🧠 **BiomedCLIP Vision Analysis**

  * Zero-shot medical image understanding
  * Anatomical and pathology-oriented classification

* 🔬 **Multi-Resolution Image Analysis**

  * Global image context
  * Multiple high-resolution local crops
  * Feature aggregation for fine-grained visual details

* 🔥 **Grad-CAM Explainability**

  * Generates visual attention heatmaps
  * Highlights regions contributing to model analysis

* 🤖 **Multi-Agent AI Architecture**

  * Triage Agent
  * Specialist Reasoning Agent
  * Interactive Clinical Q&A Agent
  * Stateful LangGraph workflow

* 📚 **Clinical Guideline RAG**

  * Retrieves relevant ACR Appropriateness Criteria
  * Grounds specialist reasoning with guideline context

* 🩻 **DICOM Support**

  * Native `.dcm` ingestion
  * PACS metadata extraction
  * Hounsfield Unit windowing

* 📄 **Clinical PDF Reports**

  * Original scan
  * AI-generated findings
  * Explainability heatmaps
  * Reasoning and disclaimers

* ⚡ **Asynchronous Backend**

  * FastAPI
  * Uvicorn
  * In-memory image processing

* 🐳 **Docker Ready**

  * Containerized deployment
  * Cloud deployment compatible

---

## 🏗️ Architecture

```text
Medical Image
     │
     ▼
┌──────────────────────────────┐
│ DICOM / Image Preprocessing  │
│ • Metadata Extraction        │
│ • HU Windowing               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Multi-Resolution Processing  │
│ • Global View                │
│ • Local Image Crops          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ BiomedCLIP Vision Engine     │
│ • Visual Features            │
│ • Clinical Prompt Analysis   │
│ • Grad-CAM                   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ LangGraph Multi-Agent Engine │
│                              │
│  Triage Agent                │
│       │                      │
│       ▼                      │
│  Specialist Agent            │
│       │                      │
│       ├── ACR RAG            │
│       ├── LLaMA 3.3 70B      │
│       └── Pydantic Validation│
│       │                      │
│       ▼                      │
│  Clinical Q&A Agent          │
│                              │
│  MemorySaver State            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Application Output           │
│ • Streamlit UI               │
│ • Clinical Report            │
│ • Grad-CAM Heatmap           │
│ • PDF Export                 │
└──────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component               | Technology                           |
| ----------------------- | ------------------------------------ |
| **Vision Model**        | Microsoft BiomedCLIP                 |
| **LLM**                 | Meta LLaMA 3.3 70B via Groq          |
| **Agent Orchestration** | LangGraph                            |
| **LLM Framework**       | LangChain Core                       |
| **Schema Validation**   | Pydantic v2                          |
| **Medical Imaging**     | PyDICOM, Pillow, NumPy               |
| **Explainability**      | PyTorch Grad-CAM / Gradient Saliency |
| **RAG**                 | ACR Appropriateness Criteria         |
| **Frontend**            | Streamlit                            |
| **Backend**             | FastAPI + Uvicorn                    |
| **PDF Generation**      | fpdf2                                |
| **Containerization**    | Docker                               |
| **Language**            | Python 3.10+                         |

---

## 📂 Project Structure

```text
Agentic-Medical-Image-Analyzer/
│
├── app/
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph_engine.py
│   │   ├── medical_agent.py
│   │   └── tools.py
│   │
│   ├── cv/
│   │   ├── __init__.py
│   │   ├── dicom_parser.py
│   │   └── feature_extractor.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   └── report_generator.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   └── guideline_engine.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── pdf_generator.py
│   │
│   └── main.py
│
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Git
* Docker *(optional)*
* Groq API key

---

### 1. Clone the Repository

```bash
git clone https://github.com/MohibAhmadButt/Agentic-Medical-Image-Analyzer.git

cd Agentic-Medical-Image-Analyzer
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

Do not commit `.env` or expose API keys publicly.

### 5. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

---

## ⚡ FastAPI Backend

Run the asynchronous backend with:

```bash
uvicorn app.main:app --reload --port 8000
```

API documentation:

```text
http://localhost:8000/docs
```

---

## 🐳 Docker

Build the image:

```bash
docker build -t medical-image-analyzer .
```

Run the container:

```bash
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your_api_key_here" \
  medical-image-analyzer
```

---

## 🩻 Supported Imaging

The application is designed to process:

* X-Ray / Radiographs
* CT
* MRI
* Panorex / Dental imaging
* DICOM (`.dcm`)
* PNG
* JPG / JPEG

---

## 🧠 Agent Workflow

The system uses a stateful multi-agent workflow:

### 1. Triage Agent

Receives the extracted visual telemetry and determines the appropriate reasoning pathway.

### 2. Specialist Reasoning Agent

Uses **LLaMA 3.3 70B** to interpret the structured vision information while incorporating retrieved clinical guideline context.

### 3. Validation Layer

Uses **Pydantic v2** to enforce structured output schemas and reduce malformed model responses.

### 4. Clinical Q&A Agent

Maintains conversational context and allows users to ask follow-up questions about the generated analysis.

### 5. Report Generation

The final structured output can be converted into a clinical-style PDF report containing the scan, visual explanations, findings, and required disclaimers.

---

## 🔥 Explainability

A major focus of the project is making multimodal AI analysis more interpretable.

The pipeline generates **Grad-CAM visual attention maps** that provide a visual representation of regions associated with the model's analysis.

```text
Input Scan
    │
    ▼
BiomedCLIP
    │
    ▼
Feature Activations
    │
    ▼
Gradient Analysis
    │
    ▼
Grad-CAM Heatmap
    │
    ▼
Visual Explanation
```

---

## 🔐 Environment Variables

| Variable       | Description                                        |
| -------------- | -------------------------------------------------- |
| `GROQ_API_KEY` | API key used to access the Groq-hosted LLaMA model |

Keep all credentials in environment variables and never hard-code secrets into source files.

Recommended `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

---

## ⚠️ Clinical Disclaimer

> **IMPORTANT**
>
> This software is an experimental AI decision-support and educational system.
>
> It is **not an FDA/CE-cleared diagnostic device** and is not intended to replace professional clinical judgment, primary radiological interpretation, or established hospital diagnostic protocols.
>
> AI-generated findings, heatmaps, recommendations, and guideline references may contain errors and must be independently reviewed and verified by a qualified medical professional.
>
> Do not use this system as the sole basis for diagnosis or treatment decisions.

---

## 📄 License

This project is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

## 👨‍💻 Author

**Mohib Ahmad Butt**

BS Artificial Intelligence
SZABIST Islamabad, Pakistan

**GitHub:**
[https://github.com/MohibAhmadButt](https://github.com/MohibAhmadButt)

**Project Repository:**
[https://github.com/MohibAhmadButt/Agentic-Medical-Image-Analyzer](https://github.com/MohibAhmadButt/Agentic-Medical-Image-Analyzer)

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**Built with Python, PyTorch, BiomedCLIP, LangGraph, LLaMA 3.3, FastAPI, and Streamlit.**
