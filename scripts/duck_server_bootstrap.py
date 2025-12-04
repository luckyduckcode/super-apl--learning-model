#!/usr/bin/env python3
"""Utility to bootstrap Duck Chat with external model config and optional reindex."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from api.duck_chat_api import DuckChatAPI, create_flask_app  # type: ignore  # noqa: E402


def _parse_env_overrides(items):
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid env override '{item}'. Use KEY=VALUE format.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch Duck Chat API with helper automation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="Path to EXTERNAL_MODEL_CONFIG JSON")
    parser.add_argument(
        "--adapter",
        help="Adapter folder or absolute path (sets DUCK_DEFAULT_ADAPTER)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host binding")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--skip-reindex", action="store_true", help="Skip rebuilding the library vector store")
    parser.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="Additional environment variables for this process",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Configure and (optionally) reindex without starting the Flask server",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug mode (auto reload)",
    )

    args = parser.parse_args(argv)

    if args.config:
        config_path = str(Path(args.config).expanduser().resolve())
        os.environ["EXTERNAL_MODEL_CONFIG"] = config_path
    if args.adapter:
        adapter_path = args.adapter
        if not os.path.isabs(adapter_path):
            adapter_path = str((REPO_ROOT / "adapters" / adapter_path).resolve())
        os.environ["DUCK_DEFAULT_ADAPTER"] = adapter_path
    for key, value in _parse_env_overrides(args.env).items():
        os.environ[key] = value

    duck = DuckChatAPI()

    if not args.skip_reindex:
        ok = duck._build_vector_store_from_library()
        status = "success" if ok else "failed"
        print(f"[Bootstrap] Library reindex {status}")
    else:
        print("[Bootstrap] Reindex skipped")

    external_source = (
        os.environ.get("EXTERNAL_MODEL_CONFIG")
        or os.environ.get("EXTERNAL_MODEL_URL")
        or os.environ.get("EXTERNAL_MODEL_EXE")
        or os.environ.get("EXTERNAL_MODEL_SO")
        or "transformers fallback"
    )
    adapter = os.environ.get("DUCK_DEFAULT_ADAPTER", "(none)")
    print(
        "[Bootstrap] Configuration:\n"
        f"  External source : {external_source}\n"
        f"  Default adapter : {adapter}\n"
        f"  Host/Port      : {args.host}:{args.port}\n"
        f"  Reindex        : {'no' if args.skip_reindex else 'yes'}"
    )

    if args.no_serve:
        print("[Bootstrap] No-serve flag set; exiting after setup.")
        return 0

    app = create_flask_app(duck)
    if not app:
        print("[Bootstrap] Flask is not installed. Please pip install flask to use the server mode.")
        return 2

    try:
        print("[Bootstrap] Starting Duck Chat server...")
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n[Bootstrap] Caught Ctrl+C; shutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
