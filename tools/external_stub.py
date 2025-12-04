#!/usr/bin/env python3
"""Lightweight CLI stub that emulates the external super-fast model.

The Duck Chat API invokes binaries with the args:
    --prompt <text>
    --max_new_tokens <int>

This stub echoes the prompt in a deterministic format so that tests can
validate the integration flow without needing the real C++ binary.
"""

import argparse
import json
import os
from datetime import datetime


def main() -> int:
    parser = argparse.ArgumentParser(description="External model stub")
    parser.add_argument("--prompt", required=True, help="Prompt text from Duck")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum tokens requested by Duck",
    )
    args = parser.parse_args()

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pid": os.getpid(),
        "max_new_tokens": args.max_new_tokens,
        "prompt_preview": args.prompt[:80],
        "response": f"[stub-response] {args.prompt[:60]} :: tokens={args.max_new_tokens}",
    }

    # The CLI contract is stdout only; Duck parses stdout as the model text.
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
