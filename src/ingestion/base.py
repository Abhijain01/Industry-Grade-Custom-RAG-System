from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Represents a full document."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    """Represents a chunk of a document."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int

class BaseParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    async def parse(self, file_path: str) -> Document:
        """Parses a file and returns a Document."""
        pass

class BaseChunker(ABC):
    """Abstract base class for chunking text."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunks a document into multiple smaller pieces."""
        pass
