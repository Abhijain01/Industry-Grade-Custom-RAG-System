import pytest
import os
import asyncio
import numpy as np
from src.retrieval.sparse_store import BM25SparseStore
from src.retrieval.hybrid import HybridRetriever
from src.vector_store.faiss_store import FAISSHNSWStore

# Mock Embedding Engine for Hybrid Testing
class MockEngine:
    async def embed_query(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)

@pytest.fixture
def test_sparse(tmp_path):
    store_path = tmp_path / "bm25.pkl"
    return BM25SparseStore(str(store_path)), str(store_path)

@pytest.mark.asyncio
async def test_sparse_add_and_search(test_sparse):
    store, _ = test_sparse
    
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "A quick brown dog"
    ]
    metas = [{"id": 0}, {"id": 1}, {"id": 2}]
    
    await store.add(texts, metas)
    
    # Needs to process before valid searching (the mock _process runs in thread)
    await asyncio.sleep(0.1) # Yield to let thread finish 
    
    res = await store.get_relevant_documents("quick dog", top_k=2)
    
    assert len(res) == 2
    # Expect "A quick brown dog" or "The quick brown fox..."
    assert res[0]["id"] in [0, 2]
    assert res[1]["id"] in [0, 2]

def test_sparse_save_load(test_sparse):
    store, p = test_sparse
    
    async def populate():
        await store.add(["hello world"], [{"id": 1}])
        await asyncio.sleep(0.1)
    asyncio.run(populate())
    
    store.save()
    assert os.path.exists(p)
    
    new_store = BM25SparseStore(p)
    new_store.load()
    
    assert len(new_store.corpus) == 1
    assert "hello" in new_store.corpus[0]

@pytest.mark.asyncio
async def test_hybrid_rrf():
    # Setup Mocks
    dense = FAISSHNSWStore(2, "d.bin", "d.json")
    sparse = BM25SparseStore("s.pkl")
    engine = MockEngine()
    
    retriever = HybridRetriever(dense, sparse, engine)
    retriever.enable_bm25 = True
    
    # Direct test of RRF method
    dense_res = [
        ({"id": "A"}, 0.9),  # Rank 1
        ({"id": "B"}, 0.8),  # Rank 2
        ({"id": "C"}, 0.7)   # Rank 3
    ]
    
    sparse_res = [
        ({"id": "B"}, 0.95), # Rank 1
        ({"id": "C"}, 0.85), # Rank 2
        ({"id": "D"}, 0.75)  # Rank 3
    ]
    
    merged = retriever._reciprocal_rank_fusion(dense_res, sparse_res)
    
    assert len(merged) == 4
    
    # B is dense rank 2, sparse rank 1 -> rrf score: (1/62) + (1/61)
    # A is dense rank 1, sparse rank None -> rrf score: (1/61)
    # B should be ranked first.
    assert merged[0]["id"] == "B"
