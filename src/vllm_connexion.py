# File: src/llm_utils.py
import requests
import time

def local_vllm_call(prompt: str, retries: int = 2) -> str:
    """VLLM call"""
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'model': './mistral_7b_4bit_bnb/',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.7,
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                'http://localhost:8000/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.Timeout as e:
            print(f"[WARN] LLM request timed out (attempt {attempt + 1})")
        except Exception as e:
            print(f"[ERROR] LLM request failed: {e}")
        
        time.sleep(2)

    return '{"days": [], "hotels": {}, "total_cost": 0}'
