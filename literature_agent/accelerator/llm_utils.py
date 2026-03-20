import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# DeepSeek API Configuration
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

def call_deepseek(prompt: str, system_prompt: str = "You are a helpful assistant.") -> dict:
    """
    Utility to call the DeepSeek API and return a JSON dictionary.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        # Fallback to simulation mode if no API key is provided, for ease of demo
        # But in real use, this would raise an error
        return None

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    
    import time
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                stream=False
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < 2: # Retry twice
                time.sleep(2 ** attempt)
                continue
            raise e
