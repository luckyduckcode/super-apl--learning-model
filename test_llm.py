#!/usr/bin/env python3
from src.api.duck_chat_api import DuckChatAPI

duck = DuckChatAPI()

print("Testing Duck with LLM-powered responses\n" + "="*60)

tests = [
    "What is the weather like today?",
    "Tell me about Python programming",
    "How do you solve differential equations?",
    "What makes a good software architect?"
]

for test in tests:
    resp = duck.get_response(test)
    print(f'\nUser: {test}')
    print(f'Duck: {resp["duck_response"]}')
    print("-" * 60)
