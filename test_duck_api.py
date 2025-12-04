#!/usr/bin/env python3
"""
Test Duck API functionality
Tests Python module, CLI, and (optionally) REST interfaces
"""

import sys
import os
import json
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'api'))

from duck_api import DuckAPI

def test_python_api():
    """Test Python module API"""
    print("\n" + "="*60)
    print("Testing Python Module API")
    print("="*60 + "\n")
    
    duck = DuckAPI()
    
    # Test 1: Matrix multiply
    print("[Test 1] Matrix Multiply")
    result = duck.matrix_multiply([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]])
    assert result['status'] == 'success', "Status should be success"
    assert result['result'] == [[4.0, 5.0], [10.0, 11.0]], "Result mismatch"
    print(f"✓ Result: {result['result']}")
    print(f"✓ Engine: {result['engine']}")
    print(f"✓ Time: {result['time_ms']:.2f}ms\n")
    
    # Test 2: Transpose
    print("[Test 2] Transpose")
    result = duck.transpose([[1, 2, 3], [4, 5, 6]])
    assert result['status'] == 'success', "Status should be success"
    assert result['result'] == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], "Result mismatch"
    print(f"✓ Result: {result['result']}\n")
    
    # Test 3: Batch processing
    print("[Test 3] Batch Processing")
    data = [
        {"A": [[1, 2, 3]], "W": [[1], [2], [3]]},
        {"A": [[4, 5, 6]], "W": [[1], [1], [1]]}
    ]
    results = duck.batch_compute(data)
    assert len(results) == 2, "Should have 2 results"
    assert results[0]['status'] == 'success', "First result should succeed"
    assert results[0]['result'] == [[14.0]], "First result mismatch"
    assert results[1]['result'] == [[15.0]], "Second result mismatch"
    print(f"✓ Processed {len(results)} items")
    for r in results:
        print(f"  Item {r['index']}: {r['result']}\n")
    
    # Test 4: Status
    print("[Test 4] API Status")
    status = duck.get_status()
    assert status['status'] == 'online', "API should be online"
    print(f"✓ API Version: {status['api_version']}")
    print(f"✓ Model: {status['model_name']}")
    print(f"✓ C++ Engine: {status['cpp_engine_available']}\n")
    
    # Test 5: Personality
    print("[Test 5] Duck Personality")
    personality = duck.get_personality()
    assert personality['status'] == 'success', "Should load personality"
    profile = personality['personality']['personality_profile']
    assert profile['humor_style'] == 'R2-D2', "Humor should be R2-D2"
    assert profile['versatility_style'] == 'C-3PO', "Versatility should be C-3PO"
    print(f"✓ Humor Style: {profile['humor_style']}")
    print(f"✓ Versatility: {profile['versatility_style']}\n")
    
    print("✅ All Python API tests passed!")
    return True


def test_cli():
    """Test CLI interface"""
    print("\n" + "="*60)
    print("Testing CLI Interface")
    print("="*60 + "\n")
    
    # Test: Get status
    print("[Test 1] CLI Status Command")
    result = subprocess.run(
        [sys.executable, 'src/api/duck_api.py', 'status'],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(__file__) or '.'
    )
    
    if result.returncode != 0:
        print(f"✗ Command failed: {result.stderr}")
        return False
    
    try:
        # Extract JSON from output (skip debug lines)
        output = result.stdout.strip()
        # Find the { to } block
        start_idx = output.find('{')
        end_idx = output.rfind('}') + 1
        
        if start_idx == -1 or end_idx <= start_idx:
            print(f"✗ No JSON output found in: {output}")
            return False
            
        json_str = output[start_idx:end_idx]
        status = json.loads(json_str)
        assert status['status'] == 'online', "API should be online"
        print(f"✓ Status: {status['status']}")
        print(f"✓ API Version: {status['api_version']}\n")
    except Exception as e:
        print(f"✗ Failed to parse output: {e}")
        print(f"  Raw output: {result.stdout}")
        return False
    
    print("✅ CLI tests passed!")
    return True


def test_rest_server():
    """Test REST API (optional - requires Flask)"""
    print("\n" + "="*60)
    print("Testing REST API Server")
    print("="*60 + "\n")
    
    try:
        import requests
    except ImportError:
        print("⚠ Flask/requests not installed. Skipping REST tests.")
        print("  Install with: pip install flask requests")
        return True
    
    from duck_api import create_flask_app, DuckAPI
    
    duck = DuckAPI()
    app = create_flask_app(duck)
    
    if not app:
        print("⚠ Flask not available. Skipping REST tests.")
        return True
    
    # Create test client
    client = app.test_client()
    
    # Test 1: Status endpoint
    print("[Test 1] GET /api/v1/status")
    response = client.get('/api/v1/status')
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'online', "API should be online"
    print(f"✓ Status: {data['status']}\n")
    
    # Test 2: Matrix multiply endpoint
    print("[Test 2] POST /api/v1/matrix-multiply")
    response = client.post(
        '/api/v1/matrix-multiply',
        json={
            'A': [[1, 2, 3], [4, 5, 6]],
            'W': [[1, 0], [0, 1], [1, 1]]
        }
    )
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should succeed"
    assert data['result'] == [[4.0, 5.0], [10.0, 11.0]], "Result mismatch"
    print(f"✓ Result: {data['result']}\n")
    
    # Test 3: Transpose endpoint
    print("[Test 3] POST /api/v1/transpose")
    response = client.post(
        '/api/v1/transpose',
        json={'A': [[1, 2, 3], [4, 5, 6]]}
    )
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should succeed"
    print(f"✓ Result: {data['result']}\n")
    
    # Test 4: Batch endpoint
    print("[Test 4] POST /api/v1/batch")
    response = client.post(
        '/api/v1/batch',
        json={
            'data': [
                {'A': [[1, 2, 3]], 'W': [[1], [2], [3]]},
                {'A': [[4, 5, 6]], 'W': [[1], [1], [1]]}
            ]
        }
    )
    assert response.status_code == 200, "Status should be 200"
    data = response.get_json()
    assert data['status'] == 'success', "Should succeed"
    assert data['batch_size'] == 2, "Should process 2 items"
    print(f"✓ Processed {data['batch_size']} items\n")
    
    # Test 5: Error handling
    print("[Test 5] Error Handling (missing W)")
    response = client.post(
        '/api/v1/matrix-multiply',
        json={'A': [[1, 2, 3]]}
    )
    assert response.status_code == 400, "Should return 400 for bad request"
    data = response.get_json()
    assert data['status'] == 'error', "Should indicate error"
    print(f"✓ Error handling works\n")
    
    print("✅ All REST API tests passed!")
    return True


def main():
    print("\n" + "="*60)
    print("Duck API Test Suite")
    print("="*60)
    
    all_passed = True
    
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
        if not test_rest_server():
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
