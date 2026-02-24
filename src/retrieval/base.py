from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class BaseRetriever(ABC):
    """Abstract base class for document retrieval."""
    
    @abstractmethod
    async def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieves and returns the most relevant documents (metadata dicts) for a query."""
        pass
