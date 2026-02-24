import pytest
import asyncio
from src.retrieval.cross_encoder import CrossEncoderReranker

@pytest.fixture
def reranker():
    # Use CPU for tests. The MiniLM cross encoder is small enough to load quickly.
    return CrossEncoderReranker(device="cpu")

@pytest.mark.asyncio
async def test_reranking_order(reranker):
    query = "What is the capital of France?"
    
    docs = [
        {"id": 1, "text": "Paris is the capital and most populous city of France."},
        {"id": 2, "text": "France is a country located in Western Europe."},
        {"id": 3, "text": "The quick brown fox jumps over the lazy dog."}
    ]
    
    reranked = await reranker.rerank(query, docs)
    
    assert len(reranked) == 3
    
    # Highly relevant should be first
    assert reranked[0]["id"] == 1
    # Tangentially relevant second
    assert reranked[1]["id"] == 2
    # Irrelevant last
    assert reranked[2]["id"] == 3
    
    # Check that score was injected
    assert "rerank_score" in reranked[0]
    assert reranked[0]["rerank_score"] > reranked[2]["rerank_score"]

@pytest.mark.asyncio
async def test_rerank_empty_list(reranker):
    query = "test"
    docs = []
    res = await reranker.rerank(query, docs)
    assert res == []

@pytest.mark.asyncio
async def test_rerank_top_k(reranker):
    query = "dogs"
    docs = [
        {"text": "I like dogs."},
        {"text": "Cats are cool."},
        {"text": "Dogs are the best pets."}
    ]
    
    res = await reranker.rerank(query, docs, top_k=2)
    assert len(res) == 2
    assert "Cats" not in res[0]["text"]
    assert "Cats" not in res[1]["text"]
