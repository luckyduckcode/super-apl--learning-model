# External Model Integration Guide

Duck Chat can delegate inference to your proprietary binaries or services with
a few environment variables or a JSON configuration file. Use whichever option
matches your deployment style.

## Quick Environment Variables

| Variable | Description |
|----------|-------------|
| `EXTERNAL_MODEL_URL` | Points to an HTTP service that exposes a `/generate` endpoint accepting `{prompt, ...}`. |
| `EXTERNAL_MODEL_EXE` | Path to a CLI binary or script that accepts `--prompt` and `--max_new_tokens` arguments. |
| `EXTERNAL_MODEL_SO`  | Path to a shared library exporting `int generate_text(char*, char*, int)`. |
| `EXTERNAL_MODEL_PYTHON` | Override the Python interpreter when the CLI path ends in `.py`. |

Set one of these before launching `duck_chat_api.py server` and Duck will skip
loading HuggingFace weights, routing every request through your handler.

## JSON Configuration (`EXTERNAL_MODEL_CONFIG`)

For more control (custom args, env, working directory), create a JSON file and
set `EXTERNAL_MODEL_CONFIG=/abs/path/to/config.json`.

```json
{
  "type": "cli",
  "path": "tools/external_stub.py",
  "python": "C:/Python312/python.exe",
  "cwd": "C:/Users/tenna/Documents/code/super apl learning model",
  "args": ["--profile", "enterprise"],
  "env": {
    "MODEL_CARD": "super-fast"
  }
}
```

Fields:
- `type`: `cli`, `http`, or `so`/`dll`.
- `path`/`exe`: Executable or shared library path (relative paths resolve from the
  config file directory).
- `args`: Extra CLI arguments inserted before Duck’s `--prompt` flag.
- `cwd`: Working directory for the process.
- `env`: Key-value map merged into the subprocess environment.
- `python`: Interpreter to use when the executable is a `.py` file.
- `url`: Required when `type` is `http`.

## Recommended Workflow

1. Place your compiled binary and any support files somewhere under `deploy/`.
2. Create a config JSON similar to the example above and commit it (without
   secrets) or generate it during deployment.
3. Export `EXTERNAL_MODEL_CONFIG` or add it to your process manager.
4. Start the server: `python src/api/duck_chat_api.py server --host 0.0.0.0 --port 5000`.

Duck will log which external model was selected when it boots, so you can verify
that the integration is active before sending traffic.
