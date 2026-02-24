import asyncio
from typing import List, Dict, Any, Optional
import os
from loguru import logger

# Import specific provider logic dynamically

from ..config import CONFIG

class QueryPipeline:
    """
    Handles Query Transformations:
    - Standard (Pass-through)
    - HyDE (Hypothetical Document Embeddings)
    - Multi-Query (Breaking down into Sub-Queries)
    """
    
    def __init__(self, mode: str = "standard", api_key: str = None):
        """
        mode: 'standard', 'hyde', or 'multi_query'
        """
        self.mode = mode.lower()
        if self.mode not in ["standard", "hyde", "multi_query"]:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        self.provider = CONFIG.generation.provider if CONFIG else "openai"
        self.model = "gemini-2.5-flash" if self.provider == "gemini" else "gpt-4o-mini"
            
        # We use a lightweight LLM for these transformations
        self.client = None
        if self.mode in ["hyde", "multi_query"]:
            if self.provider == "gemini":
                from google import genai
                
                key = api_key or os.getenv("GEMINI_API_KEY")
                if not key:
                    logger.warning("GEMINI_API_KEY not found. Query transformations will fail.")
                else:
                    self.client = genai.Client(api_key=key)
            else:
                from openai import AsyncOpenAI
                key = api_key or os.getenv("OPENAI_API_KEY")
                if not key:
                    logger.warning("OPENAI_API_KEY not found. Query transformations will fail.")
                else:
                    self.client = AsyncOpenAI(api_key=key)

    async def _generate_hyde(self, query: str) -> str:
        """Generates a hypothetical document using LLM."""
        prompt = f"Please write a passage to answer the question\nQuestion: {query}\nPassage:"
        
        try:
            if self.provider == "gemini":
                from google.genai import types
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction="You are a helpful expert assistant.",
                        temperature=0.3,
                        max_output_tokens=400
                    )
                )
                return response.text.strip()
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful expert assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query # Fallback

    async def _generate_multi_query(self, query: str) -> List[str]:
        """Generates multiple variations of the query."""
        prompt = f"""You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines.
Original question: {query}"""

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.5,
                        max_output_tokens=300
                    )
                )
                content = response.text.strip()
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=300
                )
                content = response.choices[0].message.content.strip()
            # Split by newlines and clean up numbers if listed
            # Remove leading numbers and dashes, but don't strip trailing punctuation like '?' or '.'
            import re
            
            queries = []
            for q in content.split('\n'):
                q = q.strip()
                if q:
                    # Remove leading pattern like "1. ", "- ", "2: "
                    cleaned = re.sub(r'^[\d\.\-\:\s]+', '', q)
                    queries.append(cleaned)
                    
            queries.append(query) # Always include original
            return list(set(queries))
        except Exception as e:
            logger.error(f"Multi-Query generation failed: {e}")
            return [query] # Fallback

    async def process(self, query: str) -> List[str]:
        """
        Processes the input query returning a list of strings to embed/search.
        - standard: returns [query]
        - hyde: returns [hypothetical_document]
        - multi_query: returns [query1, query2, ..., original_query]
        """
        if self.mode == "standard":
            return [query]
            
        elif self.mode == "hyde":
            if not self.client:
                return [query]
            doc = await self._generate_hyde(query)
            return [doc]
            
        elif self.mode == "multi_query":
            if not self.client:
                return [query]
            return await self._generate_multi_query(query)
