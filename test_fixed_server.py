#!/usr/bin/env python
"""Quick test script to verify fixed server and client work correctly"""

import requests
import json
import sys
import time

def test_server_api():
    """Test all server API endpoints"""
    print("\n" + "="*70)
    print("DUCK SERVER API TEST")
    print("="*70)
    
    base_url = "http://127.0.0.1:5000"
    
    try:
        # Test 1: Health check
        print("\n[1/4] Testing health endpoint...")
        r = requests.get(f"{base_url}/api/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        print("✓ Health check passed")
        
        # Test 2: Model info
        print("\n[2/4] Testing model info endpoint...")
        r = requests.get(f"{base_url}/api/model", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        print(f"✓ Model info retrieved:")
        print(f"  - Model: {data.get('model')}")
        print(f"  - Parameters: {data.get('parameters'):,}")
        print(f"  - Compression: {data.get('compression_ratio')}x")
        
        # Test 3: Status endpoint
        print("\n[3/4] Testing status endpoint...")
        r = requests.get(f"{base_url}/api/status", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        print(f"✓ Server status retrieved:")
        print(f"  - Uptime: {data.get('uptime_seconds'):.1f}s")
        print(f"  - Total inferences: {data.get('inference_count')}")
        
        # Test 4: Chat endpoint
        print("\n[4/4] Testing chat endpoint...")
        r = requests.post(
            f"{base_url}/api/chat",
            json={"message": "hello duck"},
            timeout=15
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        print(f"✓ Chat response received:")
        print(f"  - Request: {data.get('message')}")
        print(f"  - Response: {data.get('response')[:80]}...")
        print(f"  - Latency: {data.get('latency_ms'):.1f}ms")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - Server is working correctly!")
        print("="*70 + "\n")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("="*70 + "\n")
        return False

if __name__ == "__main__":
    try:
        success = test_server_api()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
