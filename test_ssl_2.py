import os
from dotenv import load_dotenv

load_dotenv()

def test_genai():
    import traceback
    try:
        from google import genai
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            print("NO KEY")
            return
            
        print("Testing standard client...")
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Hello',
        )
        print("Success:", response.text)
    except Exception as e:
        print("Standard client error:")
        traceback.print_exc()

test_genai()
