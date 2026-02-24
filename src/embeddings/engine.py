import asyncio
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer
from loguru import logger

from ..config import CONFIG

class BaseEmbeddingEngine(ABC):
    """Abstract base class for chunk embeddings."""
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a batch of texts and returns normalized vectors."""
        pass
    
    @abstractmethod
    async def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query."""
        pass

class SentenceTransformersEngine(BaseEmbeddingEngine):
    """
    Sentence Transformers implementation.
    Handles batching, async execution, vector normalization, and retries.
    """
    
    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        self.model_name = model_name or (CONFIG.embedding.model_name if CONFIG else "all-MiniLM-L6-v2")
        self.device = device or (CONFIG.embedding.device if CONFIG else "cpu")
        self.batch_size = batch_size or (CONFIG.embedding.batch_size if CONFIG else 32)
        
        logger.info(f"Loading SentenceTransformers model: {self.model_name} on {self.device}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Applies L2 normalization for cosine similarity compatibility."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Prevent division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms

    async def embed_batch(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """Embeds a batch of texts asynchronously with retry logic."""
        if not texts:
            return np.array([])
            
        loop = asyncio.get_event_loop()
        
        def _encode():
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
        for attempt in range(max_retries):
            try:
                embeddings = await loop.run_in_executor(None, _encode)
                return self._normalize(embeddings)
            except Exception as e:
                logger.error(f"Embedding failed (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt) # Exponential backoff
                
        raise RuntimeError("Embedding batch failed after all retries.")

    async def embed_query(self, text: str) -> np.ndarray:
        """Embeds a single query."""
        embeddings = await self.embed_batch([text])
        return embeddings[0]

class EmbeddingEngineFactory:
    """Factory for embedding engines."""
    
    @staticmethod
    def create(provider: str = "sentence_transformers", **kwargs) -> BaseEmbeddingEngine:
        if provider == "sentence_transformers":
            return SentenceTransformersEngine(**kwargs)
        raise ValueError(f"Unknown embedding provider: {provider}")
