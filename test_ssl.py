import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def test_genai():
    from google import genai
    from google.genai import types
    
    # Try different ways to disable SSL
    import httpx
    
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("NO KEY")
        return
        
    try:
        # Option 1: httpx client with verify=False
        custom_client = httpx.AsyncClient(verify=False)
        client = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'}) # Let's see if there's a way. Let's just create default client first
        
        # We need to test how the SDK fails
        client_default = genai.Client(api_key=key)
        response = await client_default.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents='Hello',
        )
        print("Success default:", response.text)
    except Exception as e:
        print("Error default:", str(e))

asyncio.run(test_genai())
