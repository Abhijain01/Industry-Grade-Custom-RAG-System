import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from src.retrieval.query_pipeline import QueryPipeline

# Mock response structure for OpenAI API
class MockChoice:
    def __init__(self, content):
        self.message = MagicMock()
        self.message.content = content

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

@pytest.fixture
def mock_pipeline():
    # Provide dummy key to bypass check
    pipeline = QueryPipeline(mode="multi_query", api_key="dummy")
    
    # Mock the AsyncOpenAI client
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock()
    pipeline.client = mock_client
    
    return pipeline

@pytest.mark.asyncio
async def test_standard_mode():
    pipeline = QueryPipeline(mode="standard")
    res = await pipeline.process("test query")
    assert res == ["test query"]

@pytest.mark.asyncio
async def test_hyde_mode(mock_pipeline):
    mock_pipeline.mode = "hyde"
    mock_pipeline.client.chat.completions.create.return_value = MockResponse("This is a mock hypothetical document.")
    
    res = await mock_pipeline.process("What is X?")
    assert len(res) == 1
    assert res[0] == "This is a mock hypothetical document."

@pytest.mark.asyncio
async def test_multi_query_mode(mock_pipeline):
    mock_pipeline.mode = "multi_query"
    # Provide a numbered list like an LLM might
    mock_content = "1. How does X work?\n2. Explain X.\n3. What is the function of X?"
    mock_pipeline.client.chat.completions.create.return_value = MockResponse(mock_content)
    
    original_query = "Tell me about X."
    res = await mock_pipeline.process(original_query)
    
    # Needs to extract all lines and include original
    assert len(res) == 4
    assert original_query in res
    assert "How does X work?" in res
    assert "Explain X." in res
    assert "What is the function of X?" in res

def test_invalid_mode():
    with pytest.raises(ValueError):
        QueryPipeline(mode="invalid_mode")
