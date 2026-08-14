"""
Clinical Report Generator with Pydantic Validation & Guardrails.
"""

import os
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class MedicalReport(BaseModel):
    modality: str = Field(description="e.g., Bone Radiograph / X-Ray, Chest X-Ray, Brain CT")
    primary_impression: str = Field(description="Definitive diagnostic statement of primary finding")
    key_findings: List[str] = Field(description="Bullet list of anatomical observations and visual markers")
    differential_diagnosis: List[str] = Field(description="Ranked list of alternative clinical possibilities")
    confidence_score: float = Field(description="BiomedCLIP classification confidence percentage")
    urgency_level: Literal["Routine", "Expedited", "Immediate / Emergency"] = Field(
        description="Clinical triage urgency level"
    )
    recommendations: List[str] = Field(description="Actionable clinical next steps and follow-up examinations")
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
                    "Analyze the provided zero-shot BiomedCLIP vision telemetry and clinical protocol guidelines.\n"
                    "Synthesize a strict, structured medical report following the provided schema.\n"
                    "Ensure clinical precision regarding anatomical landmarks, displacement, and urgency."
                )
            ),
            (
                "human",
                (
                    "Imaging Modality: {modality}\n"
                    "BiomedCLIP Telemetry: {telemetry}\n"
                    "Clinical Protocol Checklist: {protocol}\n\n"
                    "Generate the structured medical evaluation."
                )
            )
        ])

        self.chain = self.prompt | self.llm

    def generate(self, modality: str, telemetry: dict, protocol: dict) -> MedicalReport:
        return self.chain.invoke({
            "modality": modality,
            "telemetry": str(telemetry),
            "protocol": str(protocol)
        })
