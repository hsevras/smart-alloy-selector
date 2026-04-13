from pydantic import BaseModel, Field
from typing import Optional
from google import genai
from google.genai import types
import os
import json
import pandas as pd


class MaterialConstraints(BaseModel):
    is_3d_printable: Optional[bool] = None
    max_density_g_cm3: Optional[float] = None
    min_yield_strength_mpa: Optional[float] = None
    min_service_temp_c: Optional[float] = None
    max_cost_usd_kg: Optional[float] = None
    must_be_corrosion_resistant: Optional[bool] = None
    preferred_category: Optional[str] = None
    optimization_weights: dict = Field(default_factory=dict)


class LLMInterface:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash"


    def parse_query_to_constraints(self, query: str) -> MaterialConstraints:
        prompt = f"""
        Act as a materials scientist.

        Convert the following engineering requirement into VALID JSON ONLY.

        IMPORTANT:
        - Return only JSON
        - Use null for missing values
        - optimization_weights keys allowed:
          density, strength, cost, temperature
        - Weights must sum to 1.0

        USER QUERY:
        {query}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            ),
        )

        raw_text = response.text.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        parsed_json = json.loads(raw_text)

        return MaterialConstraints(**parsed_json)


    def generate_explanation(self, query, material_row, rejected_alternatives):
        prompt = f"""
        Explain why this material was selected.

        Query: {query}

        Selected Material:
        {material_row.to_dict()}

        Rejected Alternatives:
        {rejected_alternatives}
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        return response.text