from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

class BaseVectorStore(ABC):
    """Abstract base class for vector storage and retrieval."""
    
    @abstractmethod
    async def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Adds batched embeddings and their parallel metadata to the store."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Searches the store for the most similar vectors.
        Returns:
            - Tuple containing (distances/scores, list of metadata dicts)
        """
        pass
    
    @abstractmethod
    def save(self) -> None:
        """Persists the vector index and metadata to disk."""
        pass
        
    @abstractmethod
    def load(self) -> None:
        """Loads the vector index and metadata from disk."""
        pass
