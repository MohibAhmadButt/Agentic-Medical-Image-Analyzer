"""
Clinical Report Generator with Pydantic Schema Validation & Evidence Grounding.
Uses LLaMA 3.3 70B via Groq to synthesize structured diagnostic reports.
"""

import os
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class MedicalReport(BaseModel):
    modality: str = Field(description="Target imaging modality (e.g., Bone Radiograph / X-Ray, Chest X-Ray)")
    primary_impression: str = Field(description="Definitive diagnostic statement of primary finding")
    guideline_citation: str = Field(description="Official practice guideline cited (e.g., ACR Appropriateness Criteria)")
    key_findings: List[str] = Field(description="Bullet list of anatomical observations, fractures, and visual markers")
    differential_diagnosis: List[str] = Field(description="Ranked list of alternative clinical possibilities")
    confidence_score: float = Field(description="BiomedCLIP classification confidence percentage")
    urgency_level: Literal["Routine", "Expedited", "Immediate / Emergency"] = Field(
        description="Clinical triage urgency level"
    )
    recommendations: List[str] = Field(description="Actionable management steps and follow-ups grounded strictly in the guideline")
    disclaimer: str = Field(
        default="AI clinical decision-support utility. Requires verification by a certified medical specialist.",
        description="Mandatory medical legal disclaimer"
    )


class ClinicalReportSynthesizer:
    def __init__(self, api_key: str = None):
        groq_key = api_key or os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=groq_key,
            temperature=0.0
        ).with_structured_output(MedicalReport)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are a Senior Radiologist and Clinical Decision Support AI.\n"
                    "Analyze the provided zero-shot BiomedCLIP vision telemetry, the retrieved CLINICAL PRACTICE GUIDELINE, "
                    "and the anatomical checklist.\n"
                    "Synthesize a strict, structured medical report adhering to the provided schema.\n"
                    "Ensure clinical accuracy regarding anatomical landmarks, displacement, urgency, and cited management protocols."
                )
            ),
            (
                "human",
                (
                    "Imaging Modality: {modality}\n"
                    "BiomedCLIP Telemetry: {telemetry}\n"
                    "Retrieved Practice Guideline: {guideline}\n"
                    "Clinical Evaluation Checklist: {protocol}\n\n"
                    "Generate the structured, guideline-grounded medical report."
                )
            )
        ])

        self.chain = self.prompt | self.llm

    def generate(self, modality: str, telemetry: dict, guideline: dict, protocol: dict) -> MedicalReport:
        return self.chain.invoke({
            "modality": modality,
            "telemetry": str(telemetry),
            "guideline": str(guideline),
            "protocol": str(protocol)
        })
