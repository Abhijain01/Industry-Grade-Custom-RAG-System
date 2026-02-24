import asyncio
from typing import List, Dict, Any
from loguru import logger
from sentence_transformers import CrossEncoder

from ..config import CONFIG

class CrossEncoderReranker:
    """
    Reranks a list of retrieved documents using a Cross-Encoder.
    Cross-Encoders process the query and document simultaneously, yielding higher
    accuracy than Bi-Encoders at the cost of being slower.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        self.device = device or (CONFIG.embedding.device if CONFIG else "cpu")
        self.model_name = model_name
        
        logger.info(f"Loading CrossEncoder: {self.model_name} on {self.device}")
        self._model = CrossEncoder(
            self.model_name, 
            device=self.device, 
            automodel_args={"low_cpu_mem_usage": False}
        )
        logger.info("CrossEncoder loaded successfully.")

    async def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Reranks the document metadata dictionaries based on the query.
        Assumes each document dict has a 'text' or 'content' key containing the text to score.
        """
        if not documents:
            return []
            
        k = top_k or len(documents)
        k = min(k, len(documents))

        # We need the actual text to compare against the query
        # Support either "text" (from chunks) or "content"
        texts = [doc.get("text", doc.get("content", "")) for doc in documents]
        
        # Prepare pairs: (query, text)
        pairs = [[query, txt] for txt in texts]
        
        loop = asyncio.get_event_loop()
        
        def _score_pairs():
            # CrossEncoder returns a numpy array of scores
            return self._model.predict(pairs, show_progress_bar=False)
            
        scores = await loop.run_in_executor(None, _score_pairs)
        
        # Attach scores to documents for sorting
        scored_docs = list(zip(documents, scores))
        
        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the top k metadata dictionaries, optionally injecting the rerank score
        reranked_docs = []
        for doc, score in scored_docs[:k]:
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            reranked_docs.append(doc_copy)
            
        return reranked_docs
