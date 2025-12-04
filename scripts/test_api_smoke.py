#!/usr/bin/env python3
"""Quick smoke test for the Duck Chat API server.

Hits /api/v1/status and /api/v1/chat to confirm the real engine is responding.
"""

import sys
import requests
import json
from time import sleep

BASE_URL = "http://127.0.0.1:5000"
MAX_RETRIES = 5
RETRY_DELAY = 2


def test_status(session):
    """Test /api/v1/status endpoint."""
    try:
        resp = session.get(f"{BASE_URL}/api/v1/status", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ Status: {data.get('status')}")
            print(f"  Service: {data.get('service')}")
            print(f"  Device: {data.get('device')}")
            return True
        else:
            print(f"✗ Status returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"✗ Status failed: {e}")
        return False


def test_chat(session):
    """Test /api/v1/chat endpoint."""
    try:
        payload = {"message": "Hello, what is your name?"}
        resp = session.post(f"{BASE_URL}/api/v1/chat", json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✓ Chat: status={data.get('status')}")
            print(f"  Duck says: {data.get('duck_response', '(empty)')[:100]}...")
            return True
        else:
            print(f"✗ Chat returned {resp.status_code}: {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"✗ Chat failed: {e}")
        return False


def test_personality(session):
    """Test /api/v1/personality endpoint."""
    try:
        resp = session.get(f"{BASE_URL}/api/v1/personality", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            profile = data.get("personality", {}).get("personality_profile", {})
            print(f"✓ Personality: Humor={profile.get('humor_style')}, Versatility={profile.get('versatility_style')}")
            return True
        else:
            print(f"✗ Personality returned {resp.status_code}")
            return False
    except Exception as e:
        print(f"✗ Personality failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Duck Chat API Smoke Test")
    print("=" * 60)
    print(f"\nTarget: {BASE_URL}\n")

    # Wait for server to be ready
    print("[*] Waiting for server to be ready...")
    session = requests.Session()
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(f"{BASE_URL}/api/v1/status", timeout=2)
            if resp.status_code == 200:
                print(f"[*] Server is ready!\n")
                break
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES - 1:
                print(f"[*] Retry {attempt + 1}/{MAX_RETRIES} in {RETRY_DELAY}s...")
                sleep(RETRY_DELAY)
            else:
                print(f"[!] Server not responding after {MAX_RETRIES} attempts. Is it running?")
                return 1

    all_passed = True

    # Run tests
    print("[Test 1] GET /api/v1/status")
    if not test_status(session):
        all_passed = False
    print()

    print("[Test 2] POST /api/v1/chat")
    if not test_chat(session):
        all_passed = False
    print()

    print("[Test 3] GET /api/v1/personality")
    if not test_personality(session):
        all_passed = False
    print()

    print("=" * 60)
    if all_passed:
        print("✅ All smoke tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
