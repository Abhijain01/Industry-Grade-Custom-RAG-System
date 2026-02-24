import pytest
import asyncio
import os
from pathlib import Path
from src.ingestion.base import Document, Chunk
from src.ingestion.parsers import TextParser, ParserFactory
from src.ingestion.chunker import SlidingWindowChunker
import pydantic

# Fixture setup for temporary files
@pytest.fixture
def temp_workspace(tmp_path):
    # Create temp files
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello world! This is a test document with some content to chunk.", encoding="utf-8")
    
    md_file = tmp_path / "test.md"
    md_file.write_text("# Markdown\n\nSome more data.", encoding="utf-8")
    
    return {
        "txt": str(txt_file),
        "md": str(md_file)
    }

@pytest.mark.asyncio
async def test_text_parser(temp_workspace):
    parser = ParserFactory.get_parser(temp_workspace["txt"])
    assert isinstance(parser, TextParser)
    
    doc = await parser.parse(temp_workspace["txt"])
    
    assert isinstance(doc, Document)
    assert doc.content == "Hello world! This is a test document with some content to chunk."
    assert doc.metadata["extension"] == ".txt"
    assert "timestamp" in doc.metadata
    assert "source" in doc.metadata

def test_sliding_window_chunker():
    # Long text for chunking testing
    text = "Word " * 100 # 100 words -> ~100 tokens broadly
    
    doc = Document(content=text, metadata={"source": "test.txt", "author": "me"})
    
    # 50 tokens chunk, 10 overlap
    chunker = SlidingWindowChunker(chunk_size=50, chunk_overlap=10)
    
    chunks = chunker.chunk(doc)
    
    # Very basic validation
    assert len(chunks) > 1
    assert all(isinstance(c, Chunk) for c in chunks)
    
    # Check overlaps and metadata intactness
    assert chunks[0].metadata["source"] == "test.txt"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert "token_count" in chunks[0].metadata
    
    assert chunks[0].metadata["token_count"] <= 50

def test_chunker_overlap_error():
    with pytest.raises(ValueError):
        SlidingWindowChunker(chunk_size=50, chunk_overlap=50)

def test_pydantic_models():
    doc = Document(content="test", metadata={"foo": "bar"})
    assert doc.content == "test"
    assert doc.metadata["foo"] == "bar"

    chunk = Chunk(text="test chunk", metadata={}, chunk_index=1)
    assert chunk.text == "test chunk"
    assert chunk.chunk_index == 1
