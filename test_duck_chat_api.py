#!/usr/bin/env python3
"""
Test suite for Duck Chat API
Tests Python module, CLI, and REST interfaces
"""

import sys
import os
import json
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'api'))

from duck_chat_api import DuckChatAPI, ExternalCliModel


class _EnvPatch:
    """Context manager to temporarily patch environment variables during tests."""

    def __init__(self, **env_updates):
        self.env_updates = env_updates
        self._original = {}

    def __enter__(self):
        for key, value in self.env_updates.items():
            self._original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb):
        for key, value in self._original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_external_model_detection():
    """Validate that DuckChatAPI honors EXTERNAL_MODEL_EXE when provided."""
    print("\n" + "="*60)
    print("Testing External Model Detection")
    print("="*60 + "\n")

    stub_path = os.path.join(os.path.dirname(__file__), 'tools', 'external_stub.py')
    if not os.path.exists(stub_path):
        print("⚠ external_stub.py missing; skipping external model test")
        return True

    with _EnvPatch(EXTERNAL_MODEL_EXE=stub_path, EXTERNAL_MODEL_PYTHON=sys.executable):
        duck = DuckChatAPI()
        assert isinstance(duck.llm_pipeline, ExternalCliModel), "Should attach ExternalCliModel"
        response = duck.get_response("Hello external duck")
        assert response['status'] == 'success'
        assert 'stub-response' in response['duck_response'], "Should surface stub output"
        print("✓ EXTERNAL_MODEL_EXE honored via ExternalCliModel")

    print("✅ External model detection test passed!\n")
    return True


def test_external_model_config_file():
    """Ensure EXTERNAL_MODEL_CONFIG JSON is honored."""
    print("\n" + "="*60)
    print("Testing External Model Config File")
    print("="*60 + "\n")

    stub_path = os.path.join(os.path.dirname(__file__), 'tools', 'external_stub.py')
    if not os.path.exists(stub_path):
        print("⚠ external_stub.py missing; skipping config test")
        return True

    config_payload = {
        "type": "cli",
        "path": stub_path,
        "env": {"STUB_MODE": "config"}
    }
    tmp_file = tempfile.NamedTemporaryFile('w', delete=False, suffix='.json')
    try:
        json.dump(config_payload, tmp_file)
        tmp_file.close()
        with _EnvPatch(EXTERNAL_MODEL_CONFIG=tmp_file.name):
            duck = DuckChatAPI()
            assert isinstance(duck.llm_pipeline, ExternalCliModel), "Config should select ExternalCliModel"
            response = duck.get_response("config file test")
            assert response['status'] == 'success'
            print("✓ Config file selected ExternalCliModel")
    finally:
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass

    print("✅ External model config test passed!\n")
    return True

def test_python_api():
    """Test Python module API"""
    print("\n" + "="*60)
    print("Testing Duck Chat Python Module API")
    print("="*60 + "\n")
    
    duck = DuckChatAPI()
    
    # Test 1: Simple message
    print("[Test 1] Simple Chat Message")
    response = duck.get_response("Hello Duck")
    assert response['status'] == 'success', "Status should be success"
    assert 'duck_response' in response, "Should have duck_response"
    print(f"✓ You: Hello Duck")
    print(f"✓ Duck: {response['duck_response']}\n")
    
    # Test 2: Session management
    print("[Test 2] Session Management")
    session_id = duck.create_session("test_session")
    assert session_id == "test_session", "Session ID should match"
    print(f"✓ Created session: {session_id}")
    
    response = duck.get_response("Tell me a joke", session_id)
    assert response['session_id'] == session_id, "Session should persist"
    print(f"✓ Joke response: {response['duck_response'][:50]}...")
    print(f"✓ Message count: {response['message_count']}\n")
    
    # Test 3: Get session history
    print("[Test 3] Session History")
    session = duck.get_session(session_id)
    assert session['status'] == 'success', "Should retrieve session"
    assert len(session['session']['messages']) >= 2, "Should have messages"
    print(f"✓ Session ID: {session['session']['id']}")
    print(f"✓ Messages: {len(session['session']['messages'])}")
    print(f"✓ Created: {session['session']['created_at']}\n")
    
    # Test 4: Personality
    print("[Test 4] Personality Configuration")
    personality = duck.get_personality()
    assert personality['status'] == 'success', "Should load personality"
    profile = personality['personality']['personality_profile']
    assert profile['humor_style'] == 'R2-D2', "Humor should be R2-D2"
    assert profile['versatility_style'] == 'C-3PO', "Versatility should be C-3PO"
    print(f"✓ Humor: {profile['humor_style']}")
    print(f"✓ Versatility: {profile['versatility_style']}")
    print(f"✓ Training: Llama APL\n")
    
    # Test 5: APL question
    print("[Test 5] APL/Matrix Question")
    response = duck.get_response("How do I multiply matrices in APL?")
    assert response['status'] == 'success', "Should handle APL question"
    assert 'matrix' in response['duck_response'].lower() or 'apl' in response['duck_response'].lower(), \
        "Should reference APL/matrix"
    print(f"✓ APL Question handled")
    print(f"✓ Response: {response['duck_response'][:60]}...\n")
    
    # Test 6: List sessions
    print("[Test 6] List Sessions")
    sessions = duck.list_sessions()
    assert sessions['status'] == 'success', "Should list sessions"
    assert sessions['total_sessions'] > 0, "Should have sessions"
    print(f"✓ Total sessions: {sessions['total_sessions']}")
    for session in sessions['sessions']:
        print(f"  - {session['id']}: {session['message_count']} messages\n")
    
    # Test 7: Clear session
    print("[Test 7] Clear Session")
    temp_session = duck.create_session("temp_session")
    duck.get_response("Test message", temp_session)
    result = duck.clear_session(temp_session)
    assert result['status'] == 'success', "Should clear session"
    assert temp_session not in duck.chat_sessions, "Session should be deleted"
    print(f"✓ Session cleared: {temp_session}\n")
    
    print("✅ All Python API tests passed!")
    return True


def test_cli():
    """Test CLI interface"""
    print("\n" + "="*60)
    print("Testing Duck Chat CLI Interface")
    print("="*60 + "\n")
    
    # Test 1: Chat command
    print("[Test 1] CLI Chat Command")
    result = subprocess.run(
        [sys.executable, 'src/api/duck_chat_api.py', 'chat', 'Hello Duck'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) or '.'
    )
    
    if result.returncode != 0:
        print(f"✗ Command failed: {result.stderr}")
        return False
    
    try:
        output = result.stdout.strip()
        start_idx = output.find('{')
        end_idx = output.rfind('}') + 1
        json_str = output[start_idx:end_idx]
        response = json.loads(json_str)
        
        assert response['status'] == 'success', "Status should be success"
        assert 'duck_response' in response, "Should have response"
        print(f"✓ Chat command executed")
        print(f"✓ Response: {response['duck_response'][:50]}...\n")
    except Exception as e:
        print(f"✗ Failed to parse response: {e}")
        return False
    
    # Test 2: Status command
    print("[Test 2] CLI Status Command")
    result = subprocess.run(
        [sys.executable, 'src/api/duck_chat_api.py', 'status'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) or '.'
    )
    
    if result.returncode != 0:
        print(f"✗ Command failed: {result.stderr}")
        return False
    
    try:
        output = result.stdout.strip()
        start_idx = output.find('{')
        end_idx = output.rfind('}') + 1
        json_str = output[start_idx:end_idx]
        status = json.loads(json_str)
        
        assert status['status'] == 'online', "Service should be online"
        print(f"✓ Service: {status['service']}")
        print(f"✓ Status: {status['status']}")
        print(f"✓ Training: {status['training']}\n")
    except Exception as e:
        print(f"✗ Failed to parse status: {e}")
        return False
    
    # Test 3: Personality command
    print("[Test 3] CLI Personality Command")
    result = subprocess.run(
        [sys.executable, 'src/api/duck_chat_api.py', 'personality'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) or '.'
    )
    
    if result.returncode != 0:
        print(f"✗ Command failed: {result.stderr}")
        return False
    
    try:
        output = result.stdout.strip()
        start_idx = output.find('{')
        end_idx = output.rfind('}') + 1
        json_str = output[start_idx:end_idx]
        personality = json.loads(json_str)
        
        assert personality['status'] == 'success', "Should load personality"
        profile = personality['personality']['personality_profile']
        print(f"✓ Humor: {profile['humor_style']}")
        print(f"✓ Versatility: {profile['versatility_style']}\n")
    except Exception as e:
        print(f"✗ Failed to parse personality: {e}")
        return False
    
    print("✅ All CLI tests passed!")
    return True


def test_rest_api():
    """Test REST API"""
    print("\n" + "="*60)
    print("Testing Duck Chat REST API")
    print("="*60 + "\n")
    
    try:
        import requests
    except ImportError:
        print("⚠ requests not installed. Skipping REST tests.")
        print("  Install with: pip install requests")
        return True
    
    from duck_chat_api import create_flask_app, DuckChatAPI
    
    duck = DuckChatAPI()
    app = create_flask_app(duck)
    
    if not app:
        print("⚠ Flask not available. Skipping REST tests.")
        return True
    
    client = app.test_client()
    
    # Test 1: Chat endpoint
    print("[Test 1] POST /api/v1/chat")
    response = client.post(
        '/api/v1/chat',
        json={"message": "Hello Duck"}
    )
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should succeed"
    print(f"✓ Status: {data['status']}")
    print(f"✓ Response: {data['duck_response'][:50]}...\n")
    
    session_id = data['session_id']
    
    # Test 2: Session endpoint
    print("[Test 2] POST /api/v1/session")
    response = client.post('/api/v1/session', json={"session_id": "test_chat"})
    assert response.status_code == 201, "Status should be 201"
    data = response.get_json()
    assert data['status'] == 'success', "Should create session"
    print(f"✓ Created session: {data['session_id']}\n")
    
    # Test 3: Get session
    print("[Test 3] GET /api/v1/session/<id>")
    response = client.get(f'/api/v1/session/{session_id}')
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should retrieve session"
    assert len(data['session']['messages']) >= 1, "Should have messages"
    print(f"✓ Messages: {len(data['session']['messages'])}")
    print(f"✓ Session: {data['session']['id']}\n")
    
    # Test 4: List sessions
    print("[Test 4] GET /api/v1/sessions")
    response = client.get('/api/v1/sessions')
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should list sessions"
    print(f"✓ Total sessions: {data['total_sessions']}\n")
    
    # Test 5: Get personality
    print("[Test 5] GET /api/v1/personality")
    response = client.get('/api/v1/personality')
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should load personality"
    profile = data['personality']['personality_profile']
    print(f"✓ Humor: {profile['humor_style']}")
    print(f"✓ Versatility: {profile['versatility_style']}\n")
    
    # Test 6: Get status
    print("[Test 6] GET /api/v1/status")
    response = client.get('/api/v1/status')
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'online', "Should be online"
    print(f"✓ Service: {data['service']}")
    print(f"✓ Training: {data['training']}\n")
    
    # Test 7: Error handling
    print("[Test 7] Error Handling")
    response = client.post('/api/v1/chat', json={})
    assert response.status_code == 400, "Should return 400 for empty message"
    data = response.get_json()
    assert data['status'] == 'error', "Should indicate error"
    print(f"✓ Empty message error handled\n")
    
    # Test 8: Delete session
    print("[Test 8] DELETE /api/v1/session/<id>")
    response = client.delete(f'/api/v1/session/test_chat')
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should delete session"
    print(f"✓ Session deleted\n")
    
    print("✅ All REST API tests passed!")
    return True


def main():
    print("\n" + "="*60)
    print("Duck Chat API Test Suite")
    print("="*60)
    
    all_passed = True

    # External model detection
    try:
        if not test_external_model_detection():
            all_passed = False
    except Exception as e:
        print(f"\n❌ External model detection tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        if not test_external_model_config_file():
            all_passed = False
    except Exception as e:
        print(f"\n❌ External model config tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test Python API
    try:
        if not test_python_api():
            all_passed = False
    except Exception as e:
        print(f"\n❌ Python API tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test CLI
    try:
        if not test_cli():
            all_passed = False
    except Exception as e:
        print(f"\n❌ CLI tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Test REST
    try:
        if not test_rest_api():
            all_passed = False
    except Exception as e:
        print(f"\n❌ REST API tests failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60 + "\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
