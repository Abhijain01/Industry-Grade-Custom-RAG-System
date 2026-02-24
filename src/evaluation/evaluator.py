import asyncio
from typing import List, Dict, Any
from loguru import logger
import json
import re

from ..generation.generator import RAGGenerator
from ..config import CONFIG

class RAGEvaluator:
    """
    LLM-as-a-judge evaluator for RAG responses.
    Evaluates:
    1. Faithfulness: Is the answer derived *only* from the provided context?
    2. Relevancy: Does the retrieved context actually contain info to answer the query?
    """
    
    def __init__(self, api_key: str = None):
        # We reuse the Generator's LLM client logic for evaluation
        self.llm = RAGGenerator(api_key=api_key)
        # Force a capable model for evaluation 
        # (e.g., GPT-4o rather than mini if precision is needed, but we keep default here)
        self.llm.model = "gpt-4o" if self.llm.provider == "openai" else "claude-3-5-sonnet-20240620"
        self.llm.temperature = 0.0 # Deterministic grading

    async def _grade_faithfulness(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        prompt = f"""You are an objective grader evaluating whether an AI's answer is faithful to the provided context.
If the answer contains information that cannot be directly inferred from the context, it is unfaithful.

Context:
{context}

Question: {query}
Answer: {answer}

Grade the faithfulness from 1 to 5, where 1 means completely unfaithful/hallucinated and 5 means perfectly faithful.
Also provide a brief reason.
Output strictly in JSON format: {{"score": <int>, "reason": "<string>"}}"""

        try:
            # We construct a fake document list to reuse the generate_answer logic smoothly,
            # or we adapt it to just send the raw prompt.
            if self.llm.provider == "openai":
                response = await self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format={"type": "json_object"}
                )
                res_text = response.choices[0].message.content.strip()
            else:
                response = await self.llm.client.messages.create(
                    model=self.llm.model,
                    max_tokens=300,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}]
                )
                res_text = response.content[0].text.strip()
                
            match = re.search(r'\{.*\}', res_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return {"score": 0, "reason": "Failed to parse JSON result"}
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return {"score": 0, "reason": str(e)}

    async def evaluate(self, query: str, documents: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Runs the evaluation pipeline."""
        context_str = self.llm._build_context(documents)
        
        # Parallelize multiple evaluation metrics if we add Relevancy later
        faithfulness = await self._grade_faithfulness(query, context_str, answer)
        
        return {
            "faithfulness": faithfulness
        }
