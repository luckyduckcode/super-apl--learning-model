#!/usr/bin/env python3
"""Direct server launcher - minimal setup"""
import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("[Duck] Starting Duck Chat API server...", flush=True)
print("[Duck] Loading Flask app...", flush=True)

try:
    print("[Duck] Step 1: Importing modules...", flush=True)
    try:
        from api.duck_chat_api import DuckChatAPI, create_flask_app
        print("[Duck] Step 1: [OK] Modules imported", flush=True)
    except Exception as e:
        print(f"[Duck] Step 1 FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    
    print("[Duck] Step 2: Initializing Duck Chat API...", flush=True)
    try:
        api = DuckChatAPI()
        print("[Duck] Step 2: [OK] API initialized", flush=True)
    except Exception as e:
        print(f"[Duck] Step 2 FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    
    print("[Duck] Step 3: Creating Flask app...", flush=True)
    try:
        app = create_flask_app(api)
        print("[Duck] Step 3: [OK] Flask app created", flush=True)
    except Exception as e:
        print(f"[Duck] Step 3 FAILED: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)
    
    print("[Duck] [OK] Server ready", flush=True)
    print("[Duck] Running on http://127.0.0.1:5000", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
    
except Exception as e:
    print(f"[Duck] FATAL ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


