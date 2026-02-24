from typing import List
import tiktoken

from .base import BaseChunker, Document, Chunk
from ..config import CONFIG

class SlidingWindowChunker(BaseChunker):
    """
    Chunks a document using a sliding window approach with token-aware sizing.
    Uses tiktoken to ensure chunks fit nicely within embedding/LLM limits.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, model_name: str = "gpt-4o"):
        # Use config if not explicitly passed
        self.chunk_size = chunk_size or (CONFIG.ingestion.chunk_size if CONFIG else 512)
        self.chunk_overlap = chunk_overlap or (CONFIG.ingestion.chunk_overlap if CONFIG else 64)
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Overlap cannot be greater than or equal to chunk size")
            
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
    def chunk(self, document: Document) -> List[Chunk]:
        """Splits the document's content into overlapping token-aware chunks."""
        tokens = self.tokenizer.encode(document.content)
        
        chunks = []
        chunk_index = 0
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunk_metadata = document.metadata.copy()
            chunk_metadata["token_count"] = len(chunk_tokens)
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_index=chunk_index
            ))
            
            chunk_index += 1
            # Advance start by taking the step: size minus overlap
            step = self.chunk_size - self.chunk_overlap
            start += step

        return chunks
