import requests
import json
import os

from dotenv import load_dotenv

load_dotenv()

response = requests.post(
  url="https://openrouter.ai/api/v1/embeddings",
  headers={
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-OpenRouter-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  data=json.dumps({
    "model": "qwen/qwen3-embedding-8b",
    "input": "Your text string goes here",
    # "input": ["text1", "text2", "text3"], # batch embeddings also supported!
    "encoding_format": "float"
  })
)

print(response.json())

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": os.getenv('OPENROUTER_CHAT_MODEL'),
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
  })
)

print(response.json())