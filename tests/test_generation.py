import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.generation.generator import RAGGenerator

# Needs OPENAI fake key for instantiation without warnings causing issues
@pytest.fixture
def run_generator():
    return RAGGenerator(api_key="sk-fake")

def test_context_building(run_generator):
    docs = [
        {"filename": "doc1.txt", "text": "This is doc 1."},
        {"filename": "doc2.txt", "text": "This is doc 2."},
    ]
    
    context = run_generator._build_context(docs)
    
    assert "[Source: doc1.txt]" in context
    assert "This is doc 1." in context
    assert "[Source: doc2.txt]" in context
    assert "This is doc 2." in context

def test_token_truncation(run_generator):
    run_generator.context_token_limit = 20 # Very small limit
    
    docs = [
        {"filename": "doc1.txt", "text": "This is doc 1, taking up some tokens."},
        {"filename": "doc2.txt", "text": "This should be truncated entirely if the limit is strict enough."}
    ]
    
    context = run_generator._build_context(docs)
    assert "[Source: doc1.txt]" in context
    assert "[Source: doc2.txt]" not in context

@pytest.mark.asyncio
async def test_openai_generation(run_generator):
    mock_client = AsyncMock()
    
    class MockMessage:
        content = "This is a mocked answer."
    class MockChoice:
        message = MockMessage()
    class MockResponse:
        choices = [MockChoice()]
        
    mock_client.chat.completions.create.return_value = MockResponse()
    
    run_generator.provider = "openai"
    run_generator.client = mock_client
    
    docs = [{"filename": "doc.txt", "text": "Context text."}]
    res = await run_generator.generate_answer("query?", docs)
    
    assert res == "This is a mocked answer."
