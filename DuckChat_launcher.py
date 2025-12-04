#!/usr/bin/env python
# DuckChat wrapper - Simple launcher for the API server
import sys
import os

# Add project path
sys.path.insert(0, os.path.dirname(__file__))

if __name__ == '__main__':
    # FIX: If run without arguments (double-click), default to 'server' mode
    if len(sys.argv) == 1:
        sys.argv.append("server")

    try:
        # Use the Chat API (which includes the LLM and APL engine)
        from src.api.duck_chat_api import main_cli
        sys.exit(main_cli())
    except Exception as e:
        import traceback
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        print("\nPress Enter to exit...")
        input()
