import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch
from src.pipeline import RAGPipeline

@pytest.fixture
def run_pipeline():
    # Setup standard mocked environment
    os.environ["OPENAI_API_KEY"] = "sk-fake"
    return RAGPipeline()

@pytest.mark.asyncio
async def test_pipeline_initialization(run_pipeline):
    assert run_pipeline.retriever is not None
    assert run_pipeline.reranker is not None
    assert run_pipeline.generator is not None

@pytest.mark.asyncio
async def test_end_to_end_mock(run_pipeline):
    # Mock retrieval and generator
    run_pipeline.retrieve_context = AsyncMock(return_value=[{"text": "mocked context"}])
    run_pipeline.generator.generate_answer = AsyncMock(return_value="This is the final generated answer.")
    
    response = await run_pipeline.ask("Test query?")
    
    assert response == "This is the final generated answer."
    run_pipeline.retrieve_context.assert_called_once_with("Test query?", "standard", 5)
    run_pipeline.generator.generate_answer.assert_called_once_with("Test query?", [{"text": "mocked context"}])
