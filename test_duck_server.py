"""Test Duck Server API"""
import sys
import time
import threading
import requests

sys.path.insert(0, '.')
from duck_server import DuckServerApp

# Start server in thread
def run_server():
    app = DuckServerApp(port=5000)
    app.app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to start
print("Starting Duck Server...")
time.sleep(4)

print("\nTesting Duck Server API...\n")

try:
    # Test status
    r = requests.get('http://127.0.0.1:5000/api/status', timeout=5)
    print('✓ GET /api/status')
    data = r.json()
    print(f'  Status: {data["status"]}')
    print(f'  Model: {data["model"]}')
    print(f'  Parameters: {data["parameters"]:,}')
    print(f'  Compression: {data["compression_ratio"]:.1f}x\n')
    
    # Test chat
    r = requests.post('http://127.0.0.1:5000/api/chat', 
                      json={'message': 'hello'}, timeout=5)
    print('✓ POST /api/chat')
    data = r.json()
    response_short = data['response'][:50]
    print(f'  Message: "hello"')
    print(f'  Response: "{response_short}..."')
    print(f'  Latency: {data["latency_ms"]:.0f}ms\n')
    
    # Test model info
    r = requests.get('http://127.0.0.1:5000/api/model', timeout=5)
    print('✓ GET /api/model')
    data = r.json()
    print(f'  Architecture: {data["architecture"]}')
    print(f'  Layers: {data["layers"]}')
    print(f'  Quantization: {data["quantization_bits"]}-bit\n')
    
    print('='*70)
    print('✓ ALL TESTS PASSED - SERVER IS WORKING!')
    print('='*70)
    
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
