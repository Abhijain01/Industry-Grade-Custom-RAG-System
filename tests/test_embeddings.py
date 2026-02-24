import pytest
import numpy as np
import asyncio
from src.embeddings.engine import EmbeddingEngineFactory, SentenceTransformersEngine

@pytest.fixture
def test_engine():
    # Use CPU explicitly for fast testing
    return EmbeddingEngineFactory.create("sentence_transformers", device="cpu")

def test_engine_initialization(test_engine):
    assert isinstance(test_engine, SentenceTransformersEngine)
    assert test_engine.embedding_dim > 0

@pytest.mark.asyncio
async def test_embed_batch_shape_and_norm(test_engine):
    texts = [
        "This is the first test sentence.",
        "Here is another sentence for embedding.",
        "Short text."
    ]
    
    embeddings = await test_engine.embed_batch(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 3 # batch size
    assert embeddings.shape[1] == test_engine.embedding_dim
    
    # Check L2 norm is approximately 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5)

@pytest.mark.asyncio
async def test_embed_query(test_engine):
    text = "A specific query string."
    embedding = await test_engine.embed_query(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (test_engine.embedding_dim,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, rtol=1e-5)

@pytest.mark.asyncio
async def test_empty_batch(test_engine):
    embeddings = await test_engine.embed_batch([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.size == 0
