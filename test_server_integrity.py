#!/usr/bin/env python
"""Verify server can start without crashing"""

import sys
import os

# Test 1: Import duck_server module
print("[1/3] Testing imports...")
try:
    from duck_server import DuckServerApp
    print("✓ DuckServerApp imported successfully")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Create server instance
print("\n[2/3] Testing server instantiation...")
try:
    server = DuckServerApp(model_path='models/duck_1_58bit.pt', port=5000)
    print("✓ DuckServerApp instantiated successfully")
    print(f"  - Model loaded: {server.model is not None}")
    print(f"  - Flask app initialized: {server.app is not None}")
    print(f"  - Routes registered: {len(server.app.url_map._rules) > 0}")
except Exception as e:
    print(f"✗ Failed to instantiate: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test route handlers without running server
print("\n[3/3] Testing route handlers...")
try:
    with server.app.test_client() as client:
        # Test health endpoint
        response = client.get('/api/health')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        print(f"✓ /api/health: {response.status_code}")
        
        # Test model endpoint
        response = client.get('/api/model')
        assert response.status_code == 200
        print(f"✓ /api/model: {response.status_code}")
        
        # Test status endpoint
        response = client.get('/api/status')
        assert response.status_code == 200
        print(f"✓ /api/status: {response.status_code}")
        
        # Test root endpoint
        response = client.get('/')
        assert response.status_code == 200
        print(f"✓ /: {response.status_code}")
        
        # Test chat endpoint
        response = client.post('/api/chat', json={'message': 'hello'})
        assert response.status_code == 200
        print(f"✓ /api/chat: {response.status_code}")
        
except Exception as e:
    print(f"✗ Route test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL TESTS PASSED - Server is ready to run!")
print("="*70)
