from src.api.duck_chat_api import DuckChatAPI
from pathlib import Path
base=Path.cwd()
text=(base/'library'/'test_tutorial.txt').read_text(encoding='utf-8')
D=DuckChatAPI()
prompt=f"Generate a short factual question and answer pair about this paragraph:\n\n{text[:1400]}"
resp=D.get_response(prompt)
print('duck_resp:', resp['duck_response'])

out_path = base/'training'/'test_library_dataset.jsonl'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as fh:
    import json
    d = {'instruction': 'Question about library paragraph', 'output': resp['duck_response']}
    fh.write(json.dumps(d) + '\n')
print('wrote', str(out_path))
