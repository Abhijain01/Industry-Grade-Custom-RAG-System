import os
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from pathlib import Path
from loguru import logger

from .config import load_config, CONFIG
from .ingestion.parsers import ParserFactory
from .ingestion.chunker import SlidingWindowChunker
from .embeddings.engine import EmbeddingEngineFactory
from .vector_store.faiss_store import FAISSHNSWStore
from .retrieval.sparse_store import BM25SparseStore
from .retrieval.hybrid import HybridRetriever
from .retrieval.cross_encoder import CrossEncoderReranker
from .retrieval.query_pipeline import QueryPipeline
from .generation.generator import RAGGenerator

class RAGPipeline:
    """
    Central orchestrator for the RAG system.
    Ties together Ingestion, Embedding, Storage, Retrieval, Reranking, and Generation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        # Explicit reload of config if building system from scratch
        global CONFIG
        CONFIG = load_config(config_path)
        
        # 1. Ingestion
        self.chunker = SlidingWindowChunker()
        
        # 2. Embeddings
        self.embedding_engine = EmbeddingEngineFactory.create("sentence_transformers")
        
        # 3. Vector Stores
        self.dense_store = FAISSHNSWStore(
            dimension=self.embedding_engine.embedding_dim,
            index_path=CONFIG.vector_store.index_path,
            metadata_path=CONFIG.vector_store.metadata_path
        )
        # Attempt to load existing indexes
        self.dense_store.load()
        
        self.sparse_store = BM25SparseStore("data/bm25_store.pkl")
        self.sparse_store.load()
        
        # 4 & 5. Retrieval & Reranker
        self.retriever = HybridRetriever(self.dense_store, self.sparse_store, self.embedding_engine)
        self.reranker = CrossEncoderReranker()
        
        # 6. Query Engine
        self.query_pipeline = QueryPipeline(mode="standard") # Default, can be changed per query
        
        # 7. Generation
        self.generator = RAGGenerator()
        
        logger.info("RAG Pipeline initialized successfully.")

    async def ingest_file(self, file_path: str):
        """Processes a single file from parser -> chunker -> embedder -> stores."""
        logger.info(f"Ingesting {file_path}")
        try:
            parser = ParserFactory.get_parser(file_path)
            doc = await parser.parse(file_path)
            
            chunks = self.chunker.chunk(doc)
            texts = [c.text for c in chunks]
            # Construct metadatas with the actual text included so it can be retrieved!
            metadatas = []
            for c in chunks:
                m = c.metadata.copy()
                m["text"] = c.text
                metadatas.append(m)
            
            # Embed
            embeddings = await self.embedding_engine.embed_batch(texts)
            
            # Add to stores
            await self.dense_store.add(embeddings, metadatas)
            await self.sparse_store.add(texts, metadatas)
            
            logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")

    async def ingest_directory(self, dir_path: str, save_after: bool = True):
        """Processes all supported files in a directory concurrently."""
        path = Path(dir_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory {dir_path} does not exist.")
            
        supported_exts = set(CONFIG.ingestion.supported_extensions)
        files_to_process = [
            str(p) for p in path.rglob("*") 
            if p.is_file() and p.suffix.lower() in supported_exts
        ]
        
        logger.info(f"Found {len(files_to_process)} files to ingest.")
        
        # Run ingestion in batches to avoid memory overload
        batch_size = CONFIG.ingestion.max_workers
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            tasks = [self.ingest_file(f) for f in batch]
            await asyncio.gather(*tasks)
            
        if save_after:
            self.save_stores()

    def save_stores(self):
        """Persists dense and sparse databases to disk."""
        logger.info("Saving stores to disk...")
        self.dense_store.save()
        self.sparse_store.save()

    async def retrieve_context(self, query: str, query_mode: str = "standard", top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes the retrieval logic (Query Pip -> Hybrid -> Rerank)"""
        # 1. Transform Query
        self.query_pipeline.mode = query_mode
        expanded_queries = await self.query_pipeline.process(query)
        
        # 2. Hybrid Retrieval for all queries
        all_retrieved = []
        for q in expanded_queries:
            docs = await self.retriever.get_relevant_documents(q, top_k * 2) # Fetch extra
            all_retrieved.extend(docs)
            
        # Deduplicate metadata logically (based on source and chunk index, or full text)
        unique_docs = { doc.get("text", ""): doc for doc in all_retrieved }.values()
        
        # 3. Rerank
        reranked = await self.reranker.rerank(query, list(unique_docs), top_k=top_k)
        return reranked

    async def ask(self, query: str, query_mode: str = "standard", top_k: int = 5, stream: bool = False) -> Any:
        """
        End-to-end question answering.
        Returns the text response directly if stream=False.
        Returns an AsyncGenerator if stream=True.
        """
        logger.info(f"Processing query: '{query}' (Mode: {query_mode})")
        context_docs = await self.retrieve_context(query, query_mode, top_k)
        
        if not context_docs:
            logger.warning("No context retrieved. Proceeding to LLM with empty context.")
            
        if stream:
            return self.generator.generate_stream(query, context_docs)
        else:
            return await self.generator.generate_answer(query, context_docs)
