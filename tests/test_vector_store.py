import pytest
import os
import numpy as np
import asyncio
from src.vector_store.faiss_store import FAISSHNSWStore

@pytest.fixture
def temp_store(tmp_path):
    index_path = tmp_path / "test_index.bin"
    meta_path = tmp_path / "test_meta.json"
    
    # Dimension 4 for simple testing
    store = FAISSHNSWStore(dimension=4, index_path=str(index_path), metadata_path=str(meta_path))
    return store, str(index_path), str(meta_path)

@pytest.mark.asyncio
async def test_add_and_search(temp_store):
    store, _, _ = temp_store
    
    # Create normalized test vectors
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    meta = [{"id": 0}, {"id": 1}]
    
    await store.add(vecs, meta)
    
    assert store.index.ntotal == 2
    assert len(store.metadata) == 2
    
    # Query exact match for vec 1
    q = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    distances, retrieved_meta = await store.search(q, top_k=1)
    
    assert len(retrieved_meta) == 1
    assert retrieved_meta[0]["id"] == 1
    
    # For L2 distance on exact match, distance should be 0
    assert np.isclose(distances[0], 0.0, atol=1e-5)

def test_save_and_load(temp_store):
    store, ip, mp = temp_store
    
    # Sync operation wrapper for async add
    async def populate():
        vecs = np.random.rand(5, 4).astype(np.float32)
        meta = [{"doc_id": i} for i in range(5)]
        await store.add(vecs, meta)
        
    asyncio.run(populate())
    
    # Save to disk
    store.save()
    
    assert os.path.exists(ip)
    assert os.path.exists(mp)
    
    # Create new instance and load
    new_store = FAISSHNSWStore(dimension=4, index_path=ip, metadata_path=mp)
    new_store.load()
    
    assert new_store.index.ntotal == 5
    assert len(new_store.metadata) == 5
    assert new_store.metadata[3]["doc_id"] == 3

@pytest.mark.asyncio
async def test_dimension_mismatch(temp_store):
    store, _, _ = temp_store
    
    vecs = np.array([[1.0, 0.0]], dtype=np.float32) # Dim 2, store is Dim 4
    meta = [{"id": 0}]
    
    with pytest.raises(ValueError, match="does not match index"):
        await store.add(vecs, meta)

@pytest.mark.asyncio
async def test_count_mismatch(temp_store):
    store, _, _ = temp_store
    
    vecs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    meta = [{"id": 0}, {"id": 1}] # 2 metas, 1 vec
    
    with pytest.raises(ValueError, match="does not match metadata count"):
        await store.add(vecs, meta)
