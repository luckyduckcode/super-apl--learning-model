#!/usr/bin/env python3
from src.api.duck_chat_api import DuckChatAPI

duck = DuckChatAPI()
tests = ['5+3+2+1', '5+5', '10*2', '100/4', '2*3+4']

print("Testing Math Calculations in Duck Chat API\n" + "="*50)
for test in tests:
    resp = duck.get_response(test)
    print(f'\nInput: {test}')
    print(f'Response:\n{resp["duck_response"]}')
    print("-" * 50)
