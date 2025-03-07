from typing import List
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import logging
import json

from services.llm_factory import LLMFactory

import re

def extract_json(response: str) -> str:
    """
    Extrait uniquement la partie JSON d'une réponse contenant du texte inutile avant et après.
    """
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return match.group(0).strip()  # Retourne seulement le JSON
    return ""
class SynthesizedResponse(BaseModel):
    thought_process: List[str] = Field(
        description="List of thoughts that the AI assistant had while synthesizing the answer"
    )
    answer: str = Field(description="The synthesized answer to the user's question")
    enough_context: bool = Field(
        description="Whether the assistant has enough context to answer the question"
    )


class Synthesizer:
    SYSTEM_PROMPT = """
    # Role and Purpose
    You are an AI assistant for an e-commerce FAQ system. Your task is to synthesize a coherent and helpful answer 
    based on the given question and relevant context retrieved from a knowledge database.
    
    # Response Format:
    You MUST return the response in a valid JSON format following this structure:
    {
      "thought_process": ["step 1", "step 2", "step 3"],
      "answer": "Your final answer here",
      "enough_context": true  # or false
    }
    
    # Guidelines:
    1. Provide a clear and concise answer to the question.
    2. Use only the information from the relevant context to support your answer.
    3. The context is retrieved based on cosine similarity, so some information might be missing or irrelevant.
    4. Be transparent when there is insufficient information to fully answer the question.
    5. Do not make up or infer information not present in the provided context.
    6. If you cannot answer the question based on the given context, clearly state that.
    7. Maintain a helpful and professional tone appropriate for customer service.
    8. Adhere strictly to company guidelines and policies by using only the provided knowledge base.
    """


    @staticmethod
    def generate_response(question: str, context: pd.DataFrame) -> SynthesizedResponse:
        """Generates a synthesized response based on the question and context.

        Args:
            question: The user's question.
            context: The relevant context retrieved from the knowledge base.

        Returns:
            A SynthesizedResponse containing thought process and answer.
        """
        context_str = Synthesizer.dataframe_to_json(
            context, columns_to_keep=["content", "category"]
        )

        messages = [
            {"role": "system", "content": Synthesizer.SYSTEM_PROMPT},
            {"role": "user", "content": f"# User question:\n{question}"},
            {
                "role": "assistant",
                "content": f"# Retrieved information:\n{context_str}",
            },
        ]

        llm = LLMFactory()
        raw_response = llm.create_completion(response_model=None, messages=messages)

        # 🛠️ Tentative de parsing en JSON pour s'adapter au modèle Pydantic
        try:
            cleaned_response = extract_json(raw_response)
            json_response = json.loads(cleaned_response)  # Convertit la réponse en JSON
            parsed_response = SynthesizedResponse.parse_obj(json_response)  # Vérifie la structure
            return parsed_response
        except (json.JSONDecodeError, ValidationError) as e:
            logging.error(f"❌ Error parsing response: {e}")
            return SynthesizedResponse(
                thought_process=["Error processing response"],
                answer="Je n'ai pas pu comprendre la réponse du modèle.",
                enough_context=False
            )

    @staticmethod
    def dataframe_to_json(
            context: pd.DataFrame,
            columns_to_keep: List[str],
    ) -> str:
        """
        Convert the context DataFrame to a JSON string.

        Args:
            context (pd.DataFrame): The context DataFrame.
            columns_to_keep (List[str]): The columns to include in the output.

        Returns:
            str: A JSON string representation of the selected columns.
        """
        return context[columns_to_keep].to_json(orient="records", indent=2)
