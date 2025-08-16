import requests
import os

input = "Smile"

url = "http://localhost:11434/api/generate"
data = {
    "model": "gpt-oss:20b",
    "prompt": input,
    "stream": False,
    "raw": True,
    "options": {
        "temperature": 0.7,
        "num_predict": 4000,
        "num_ctx": 2100,
        "top_p": 0.9
    }
}

response = requests.post(url, json=data)
result = response.json()

with open("input_text_for_embedding.txt", "a") as file:
    file.write(f"Input: {input}\n")
    file.write(f"Output: {result['response']}\n")
    file.write("-" * 50 + "\n")
    
print(result["response"])