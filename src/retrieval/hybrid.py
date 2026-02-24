import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from loguru import logger

from .base import BaseRetriever
from .sparse_store import BM25SparseStore
from ..vector_store.faiss_store import FAISSHNSWStore
from ..embeddings.engine import BaseEmbeddingEngine
from ..config import CONFIG

class HybridRetriever(BaseRetriever):
    """
    Combines dense retrieval (FAISS) and sparse retrieval (BM25)
    using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self, dense_store: FAISSHNSWStore, sparse_store: BM25SparseStore, embedding_engine: BaseEmbeddingEngine):
        self.dense_store = dense_store
        self.sparse_store = sparse_store
        self.embedding_engine = embedding_engine
        
        self.top_k = CONFIG.retrieval.top_k if CONFIG else 5
        self.enable_bm25 = CONFIG.retrieval.enable_bm25 if CONFIG else True
        self.rrf_k = 60 # Standard RRF constant

    def _normalize_scores(self, scores: List[float], invert_distance: bool = False) -> List[float]:
        """Min-max normalizes a list of scores."""
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        if min_s == max_s:
            return [1.0 for _ in scores]
            
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
        
        if invert_distance:
            # L2 distance: lower is better. Invert so higher is better for RRF.
            normalized = [1.0 - s for s in normalized]
            
        return normalized

    def _reciprocal_rank_fusion(self, dense_results: List[Tuple[Dict, float]], sparse_results: List[Tuple[Dict, float]]) -> List[Dict]:
        """Merges results using RRF. Requires dicts to be hashable or have unique 'doc_id'."""
        
        # We assume metadata has a unique identifier, e.g., 'chunk_id' or 'id'.
        # For this implementation, we hash a tuple of the items if 'id' isn't strictly there.
        def _get_id(meta: Dict) -> str:
            return meta.get("id", str(hash(frozenset(meta.items()))))

        rrf_scores = {}
        meta_lookup = {}
        
        # Process Dense
        for rank, (meta, score) in enumerate(dense_results):
            doc_id = _get_id(meta)
            meta_lookup[doc_id] = meta
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Process Sparse
        for rank, (meta, score) in enumerate(sparse_results):
            doc_id = _get_id(meta)
            meta_lookup[doc_id] = meta
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Sort by fused score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return merged metadata
        return [meta_lookup[doc_id] for doc_id in sorted_ids]

    async def get_relevant_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Retrieves and fuses documents."""
        k = top_k or self.top_k
        
        async def fetch_dense():
            # Generate query embedding
            q_emb = await self.embedding_engine.embed_query(query)
            dists, metas = await self.dense_store.search(q_emb, k * 2) # Fetch extra for fusion
            
            # Normalize and format
            norm_scores = self._normalize_scores(dists.tolist(), invert_distance=True)
            return list(zip(metas, norm_scores))

        async def fetch_sparse():
            if not self.enable_bm25:
                return []
            scores, metas = await self.sparse_store.search(query, k * 2)
            norm_scores = self._normalize_scores(scores)
            return list(zip(metas, norm_scores))
            
        # Run searches in parallel
        dense_results, sparse_results = await asyncio.gather(fetch_dense(), fetch_sparse())
        
        # Merge
        if not self.enable_bm25:
            return [m for m, _ in dense_results][:k]
            
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:k]
