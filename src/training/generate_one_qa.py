"""
Generate a small set of synthetic Q/A pairs by directly calling DuckChatAPI (no HTTP required).

Usage:
python src/training/generate_one_qa.py --count 5
"""

import os
import json
import argparse
import sys

# Ensure 'src' is on the path so imports like 'src.api...' work when running as a script
base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Insert the repository root (not the src folder) so 'src' can be imported as a package
sys.path.insert(0, base)
from src.api.duck_chat_api import DuckChatAPI


def generate(n=5, out='training/library_dataset.jsonl'):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    library_path = os.path.join(base, 'library')
    if not os.path.exists(library_path):
        print('No library folder found')
        return 0

    d = DuckChatAPI()
    items = []
    for i, filename in enumerate(os.listdir(library_path)):
        if i >= n:
            break
        if not filename.lower().endswith(('.txt', '.md')):
            continue
        path = os.path.join(library_path, filename)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # Query the model to generate a question from text
        prompt = f"Generate a short, factual question about this paragraph and then provide its concise answer.\n\nParagraph:\n{text[:1000]}"
        res = d.get_response(prompt)
        ans = res.get('duck_response', '').strip()
        items.append({"instruction": f"Q about {filename}", "output": ans})

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + '\n')
    print(f'Wrote {len(items)} items to {out}')
    return len(items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--out', default='training/library_dataset.jsonl')
    args = parser.parse_args()
    generate(args.count, args.out)
