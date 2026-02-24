import os
import asyncio
from typing import List, Dict, Any, AsyncGenerator
from loguru import logger
import tiktoken

from ..config import CONFIG

class RAGGenerator:
    """
    Handles prompt construction, token budgeting, and LLM generation.
    Supports both OpenAI and Anthropic dynamically based on configuration.
    """
    
    def __init__(self, api_key: str = None):
        self.provider = CONFIG.generation.provider if CONFIG else "openai"
        self.model = CONFIG.generation.model if CONFIG else "gpt-4o-mini"
        self.temperature = CONFIG.generation.temperature if CONFIG else 0.1
        self.max_tokens = CONFIG.generation.max_tokens if CONFIG else 1024
        
        # We explicitly track context token budget. Assuming around 128k for modern models, 
        # but let's be conservative to leave room for output and prompt wrappers.
        self.context_token_limit = 100_000 
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.client = None
        self.api_key = api_key
        # Delaying client setup to the actual generation method 
        # to ensure it binds to the active event loop (important for Streamlit)

    def _setup_client(self, api_key: str = None):
        if self.provider == "openai":
            from openai import AsyncOpenAI
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                logger.warning("OPENAI_API_KEY missing.")
            self.client = AsyncOpenAI(api_key=key)
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                logger.warning("ANTHROPIC_API_KEY missing.")
            self.client = AsyncAnthropic(api_key=key)
        elif self.provider == "gemini":
            from google import genai
            key = api_key or os.getenv("GEMINI_API_KEY")
            if not key:
                logger.warning("GEMINI_API_KEY missing.")
                
            self.client = genai.Client(api_key=key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Constructs context from retrieved documents, enforcing token limits.
        """
        context_parts = []
        current_tokens = 0
        
        for i, doc in enumerate(documents):
            source = doc.get("filename", doc.get("source", f"Doc_{i+1}"))
            text = doc.get("text", doc.get("content", ""))
            
            # Format with citation metadata
            chunk_str = f"[Source: {source}]\n{text}\n\n"
            chunk_tokens = self._count_tokens(chunk_str)
            
            if current_tokens + chunk_tokens > self.context_token_limit:
                logger.warning(f"Context token limit ({self.context_token_limit}) reached at document {i}. Truncating remainder.")
                break
                
            context_parts.append(chunk_str)
            current_tokens += chunk_tokens
            
        return "".join(context_parts)

    def _build_prompt(self, query: str, context_str: str) -> str:
        return f"""You are a helpful and accurate assistant. 
Use the following provided context to answer the user's question. 
If the answer is not contained within the context, simply state that you do not know based on the provided information. 
Always try to cite your sources using the [Source: filename] tags provided in the context.

Context:
{context_str}

Question: {query}
Answer:"""

    async def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generates a non-streaming answer."""
        # Ensure client is bound to the currently executing asyncio event loop
        self._setup_client(self.api_key)
        
        if not self.client:
            raise RuntimeError(f"Client for {self.provider} is not initialized properly.")
            
        context_str = self._build_context(documents)
        prompt = self._build_prompt(query, context_str)
        
        try:
            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "anthropic":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
            elif self.provider == "gemini":
                from google.genai import types
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e

    async def generate_stream(self, query: str, documents: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """Generates a streaming answer."""
        # Ensure client is bound to the currently executing asyncio event loop
        self._setup_client(self.api_key)
        
        if not self.client:
            raise RuntimeError(f"Client for {self.provider} is not initialized properly.")
            
        context_str = self._build_context(documents)
        prompt = self._build_prompt(query, context_str)
        
        try:
            if self.provider == "openai":
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
            elif self.provider == "anthropic":
                async with self.client.messages.stream(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
                        
            elif self.provider == "gemini":
                from google.genai import types
                response_stream = await self.client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                async for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise e
