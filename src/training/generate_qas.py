"""
Generate synthetic Q/A pairs from library chunks.

This script uses the running Duck LLM (or any model) to create Q/A pairs by prompting it to summarize or create QA pairs from a paragraph. It writes a JSONL file with {"instruction":..., "output":...}

Usage:
- Start your server with a model available or run as a script against the local model instance.
"""

import os
import json
import requests

LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'library')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'training', 'library_dataset.jsonl')

API_CHAT = "http://localhost:5000/api/v1/chat"


def chunk_texts(chunk_size=1500):
    for filename in os.listdir(LIB_DIR):
        if not filename.lower().endswith(('.txt', '.md')):
            continue
        path = os.path.join(LIB_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        start = 0
        while start < len(text):
            chunk = text[start:start+chunk_size]
            yield filename, chunk
            start += chunk_size


def create_qa_from_chunk(chunk_text, sample_question_prompt=None):
    # Use Duck to generate a summary + question+ answer pair
    prompt = f"Create a short, factual question and answer pair about the following paragraph. Return only JSON with keys instruction and output.\n\n{chunk_text}"
    payload = {"message": prompt}
    try:
        r = requests.post(API_CHAT, json=payload, timeout=30)
        if r.status_code == 200:
            resp = r.json()
            # the 'duck_response' is text; we can assume instructions
            out = resp.get('duck_response', '').strip()
            # Try to detect a Q/A pattern in the response; if none, wrap as instruction->summary
            return {"instruction": "Answer the question about the paragraph", "output": out}
        else:
            return None
    except Exception as e:
        print('API call failed:', e)
        return None


if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    count = 0
    with open(OUTPUT, 'w', encoding='utf-8') as outfh:
        for filename, chunk in chunk_texts():
            qa = create_qa_from_chunk(chunk)
            if qa:
                outfh.write(json.dumps(qa, ensure_ascii=False) + '\n')
                count += 1
                if count % 10 == 0:
                    print(f'Wrote {count} pairs so far...')
    print(f'Done: wrote {count} synthetic QA pairs to {OUTPUT}')
