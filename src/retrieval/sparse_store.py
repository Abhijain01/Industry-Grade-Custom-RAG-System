import os
import math
import pickle
import asyncio
from typing import List, Dict, Any, Tuple
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi

from .base import BaseRetriever

class BM25SparseStore(BaseRetriever):
    """
    Sparse Retriever using BM25.
    Tokenizes text by lowercasing and splitting by whitespace.
    For production, consider a more robust tokenizer like spacy or nltk.
    """
    def __init__(self, store_path: str = "data/bm25_store.pkl"):
        self.store_path = store_path
        self.bm25: BM25Okapi = None
        self.corpus: List[List[str]] = []
        self.metadata: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer."""
        return text.lower().split()

    async def add(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Adds documents to the BM25 corpus."""
        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadata lengths do not match.")
            
        loop = asyncio.get_event_loop()
        
        def _process():
            tokenized = [self._tokenize(t) for t in texts]
            self.corpus.extend(tokenized)
            self.metadata.extend(metadatas)
            
            # Rebuild index (rank_bm25 doesn't support incremental easily)
            self.bm25 = BM25Okapi(self.corpus)
            
        await loop.run_in_executor(None, _process)

    async def get_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Returns the top_k matching document metadata."""
        scores, metas = await self.search(query, top_k)
        return metas

    async def search(self, query: str, top_k: int = 5) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Returns BM25 scores and metadata dictionaries."""
        if not self.bm25 or not self.corpus:
            return [], []
            
        loop = asyncio.get_event_loop()
        
        def _score():
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top k indices
            top_n = min(top_k, len(scores))
            # argsort descending
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
            
            top_scores = [scores[i] for i in top_indices]
            top_metas = [self.metadata[i] for i in top_indices]
            return top_scores, top_metas
            
        return await loop.run_in_executor(None, _score)
        
    def save(self) -> None:
        """Saves corpus, metadata, and bm25 instance."""
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
        store_data = {
            "corpus": self.corpus,
            "metadata": self.metadata,
        }
        with open(self.store_path, 'wb') as f:
            pickle.dump(store_data, f)
        logger.info(f"Saved BM25 store to {self.store_path} ({len(self.corpus)} docs)")

    def load(self) -> None:
        """Loads corpus, metadata, and rebuilds bm25 instance."""
        if not os.path.exists(self.store_path):
            logger.warning(f"BM25 store file {self.store_path} not found.")
            return
            
        with open(self.store_path, 'rb') as f:
            store_data = pickle.load(f)
            
        self.corpus = store_data["corpus"]
        self.metadata = store_data["metadata"]
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            
        logger.info(f"Loaded BM25 store from {self.store_path} ({len(self.corpus)} docs)")
