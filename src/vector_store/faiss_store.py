import os
import json
import asyncio
from pathlib import Path
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from loguru import logger

from .base import BaseVectorStore
from ..config import CONFIG

class FAISSHNSWStore(BaseVectorStore):
    """
    Vector store using FAISS HNSW flat index.
    Maintains a parallel list of metadata dictionaries.
    Assumes vectors are L2-normalized externally if cosine metric is intended.
    """
    
    def __init__(self, dimension: int, index_path: str = None, metadata_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or (CONFIG.vector_store.index_path if CONFIG else "data/faiss_index.bin")
        self.metadata_path = metadata_path or (CONFIG.vector_store.metadata_path if CONFIG else "data/metadata.json")
        
        # M is number of connections per layer in HNSW (default 32 is a good balance)
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        
        # Determine metric type (Inner Product works as Cosine Similarity if L2 normalized)
        metric = CONFIG.vector_store.metric if CONFIG else "cosine"
        if metric == "cosine":
            # faiss.IndexHNSWFlat inherently uses L2 dist. 
            # To get Inner Product (Cosine sim for normalized vectors), we wrap it or just use L2 which ranks normalized vectors the same way.
            # However, IndexHNSWFlat is hardcoded to L2 dist in Python binding mostly.
            # L2 squared distance on normalized vectors = 2 - 2 * CosineSimilarity
            # So sorting by ascending L2 dist is identical to descending CosineSim.
            self.index.metric_type = faiss.METRIC_L2
            
        self.metadata: List[Dict[str, Any]] = []

    async def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Adds batched embeddings and metadata asynchronously."""
        if embeddings.shape[0] != len(metadatas):
            raise ValueError(f"Embeddings count ({embeddings.shape[0]}) does not match metadata count ({len(metadatas)})")
            
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index {self.dimension}")

        loop = asyncio.get_event_loop()
        
        def _add_to_index():
            # Ensure float32 (FAISS requirement)
            if embeddings.dtype != np.float32:
                float_embeddings = embeddings.astype(np.float32)
            else:
                float_embeddings = embeddings
                
            self.index.add(float_embeddings)
            
        await loop.run_in_executor(None, _add_to_index)
        self.metadata.extend(metadatas)

    async def search(self, query_embedding: np.ndarray, top_k: int = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Searches the store. Returns distances and corresponding metadata list."""
        if self.index.ntotal == 0:
            return np.array([]), []
            
        k = top_k or (CONFIG.retrieval.top_k if CONFIG else 5)
        k = min(k, self.index.ntotal)
        
        # Ensure 2D and float32
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        loop = asyncio.get_event_loop()
        
        def _search_index():
            distances, indices = self.index.search(query_embedding, k)
            return distances[0], indices[0]
            
        distances, indices = await loop.run_in_executor(None, _search_index)
        
        retrieved_metadata = [self.metadata[idx] for idx in indices if idx != -1]
        
        return distances, retrieved_metadata

    def save(self) -> None:
        """Synchronously persists index and metadata to disk."""
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save Metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved FAISS index to {self.index_path} ({self.index.ntotal} vectors)")

    def load(self) -> None:
        """Synchronously loads index and metadata from disk."""
        if not os.path.exists(self.index_path):
            logger.warning(f"Index file {self.index_path} not found. Skipping load.")
            return
            
        if not os.path.exists(self.metadata_path):
            logger.warning(f"Metadata file {self.metadata_path} not found. Skipping load.")
            return
            
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load Metadata
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            
        if self.index.ntotal != len(self.metadata):
            logger.error(f"Mismatch: {self.index.ntotal} vectors but {len(self.metadata)} metadata entries.")
            raise ValueError("Index and metadata out of sync on disk.")
            
        logger.info(f"Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")
