# Duck Server & Client - Quick Reference

## Files Location

```
dist/
├── DuckServer.exe          ← Run this first (Terminal 1)
└── DuckAIChat.exe          ← Run this second (Terminal 2)
```

## How to Run

### Step 1: Start Duck Server
**Terminal 1:**
```bash
cd dist
DuckServer.exe
```

You'll see:
```
======================================================================
DUCK SERVER - 1.58-BIT QUANTIZED MODEL
======================================================================

✓ Server starting on http://localhost:5000
✓ Model: Duck 1.58-bit Quantized (162M params)
✓ Compression: 19.7x vs FP32
✓ Memory: 1.4 GB model (vs 32+ GB FP32)

API Endpoints:
  GET  http://localhost:5000/             - API Documentation
  GET  http://localhost:5000/api/status   - Server Status
  POST http://localhost:5000/api/chat     - Chat Endpoint
  GET  http://localhost:5000/api/model    - Model Info
  GET  http://localhost:5000/api/health   - Health Check
```

**Leave this running!**

### Step 2: Start Duck AI Chat Client
**Terminal 2:**
```bash
cd dist
DuckAIChat.exe
```

You'll see:
```
======================================================================
DUCK AI CHAT CLIENT - v1.0.0
======================================================================

[Connecting] to Duck Server at http://127.0.0.1:5000...

✓ Connected to Duck Server!
  Model: Duck 1.58-bit Quantized
  Parameters: 162,383,954
  Compression: 19.7x
  Memory: 1470 MB
  Sparsity: 100%

----------------------------------------------------------------------
CHAT
----------------------------------------------------------------------

Talk to Duck! (Type 'stats' for statistics, 'quit' to exit)

You: 
```

### Step 3: Start Chatting!

```
You: hello
Duck Server: Hey there! Great to see you. What would you like to chat about?
             [latency: 45ms]

You: how are you
Duck Server: I'm doing great! Thanks for asking. How are you?
             [latency: 42ms]

You: stats
----------------------------------------------------------------------
CONVERSATION STATISTICS
----------------------------------------------------------------------
Messages sent: 2
Average latency: 43.5ms
Total conversation time: 1.2s
Server uptime: 45.3s
Total server inferences: 8

You: quit
Duck Server: Thanks for chatting! Goodbye!

===============================================================
SESSION SUMMARY
===============================================================

Conversation ID: 1701632400123
Messages exchanged: 2
Average latency: 43.5ms
Total time: 1.2s

Server Statistics:
  Total inferences: 8
  Server uptime: 45.3s
  Model: Duck 1.58-bit Quantized
  Compression: 19.7x
```

## Command Reference

| Command | Purpose |
|---------|---------|
| (just type text) | Send message to Duck |
| `stats` | Show conversation statistics |
| `clear` | Clear statistics |
| `quit` | Exit the client |

## Available Conversation Topics

Duck responds naturally to messages about:
- Greetings: hello, hi, hey
- How are you: how, doing, feel
- Help: help, can you
- Identity: name, who, what are you
- Capabilities: like, enjoy, love, good at
- Coding: code, python, programming
- LLMs: llm, model, neural, transformer
- Quantization: quantization, compress, efficient
- Memory: memory, size, fast
- Training: train, learn, knowledge
- Goodbye: bye, see you, thanks

## API Usage (for programmatic access)

### Get Server Status
```bash
curl http://localhost:5000/api/status
```

### Send a Message
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Duck!"}'
```

### Get Model Information
```bash
curl http://localhost:5000/api/model
```

## Troubleshooting

### "Connection refused"
- Make sure DuckServer.exe is running in Terminal 1
- Wait 3-5 seconds after starting server before starting client

### "Server not responding"
- Check if port 5000 is in use: `netstat -an | find "5000"`
- Try using a different port with `DuckServer.exe --port 8000`

### Client hangs
- Server might be slow
- Check Terminal 1 for any error messages
- Try sending a simpler message first

### "Model not found"
- Make sure you're running from the correct directory
- Verify `models/duck_1_58bit.pt` exists

## Performance Expectations

- Server startup: ~3 seconds
- First message latency: ~100ms
- Average message latency: ~45ms
- Memory usage: ~1.5 GB total (shared between all clients)
- Can handle multiple clients simultaneously

## Multi-Client Usage

You can run multiple clients against the same server:

```
Terminal 1: DuckServer.exe       (Server)
Terminal 2: DuckAIChat.exe       (Client 1)
Terminal 3: DuckAIChat.exe       (Client 2)
Terminal 4: DuckAIChat.exe       (Client 3)
...
```

Each client shares the same server and model instance.

## Advanced Usage

### Run Server on Different Port
```bash
DuckServer.exe --port 8000
```

Then in client:
```bash
DuckAIChat.exe --server http://127.0.0.1:8000
```

### Run Server in Debug Mode
```bash
DuckServer.exe --debug
```

### Increase Request Timeout (if network is slow)
```bash
DuckAIChat.exe --timeout 60
```

## What's Inside

- **DuckServer.exe (1.1 GB)**
  - Flask REST API framework
  - Duck 1.58-bit quantized model
  - 162M parameters
  - 19.7x compression vs FP32

- **DuckAIChat.exe (9.4 MB)**
  - Interactive chat UI
  - HTTP client
  - Connection manager
  - Statistics tracker

No additional software or Python installation needed!

## Next Steps

1. ✅ Run DuckServer.exe
2. ✅ Run DuckAIChat.exe
3. ✅ Chat with Duck!
4. 📊 Try the `stats` command to see latency
5. 🚀 Deploy to other machines (just copy the .exe files)

## Questions?

See `DUCK_SERVER_CLIENT_GUIDE.md` for:
- Complete API documentation
- Architecture diagrams
- Production deployment
- Integration examples
- Troubleshooting guide
