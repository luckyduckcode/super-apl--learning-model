# Duck Chat API Documentation

**Version:** 1.0.0  
**Status:** Production Ready ✅

## Overview

The Duck Chat API provides a complete conversational interface for Duck with three access methods:
- **Python Module** - Direct import and function calls
- **CLI** - Command-line interface
- **REST API** - Web service with HTTP endpoints

Duck combines R2-D2 humor with C-3PO versatility, trained on Llama APL for intelligent conversation about matrices and APL programming.

## Quick Start

### Executable (Windows)
```bash
# Start REST API server on localhost:5000
dist/DuckChat.exe

# Or with custom port
dist/DuckChat.exe server --port 8080
```

### Python Module
```python
from src.api.duck_chat_api import DuckChatAPI

duck = DuckChatAPI()
response = duck.get_response("Hello Duck, tell me a joke")
print(response['duck_response'])
```

### CLI
```bash
python src/api/duck_chat_api.py chat "Hello Duck"
python src/api/duck_chat_api.py chat "Help with matrix multiply" --session session_1
python src/api/duck_chat_api.py server --port 5000
```

### REST API
```bash
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello Duck"}'
```

## Python Module API

### DuckChatAPI Class

#### `__init__()`
Initialize the Duck Chat API.

```python
duck = DuckChatAPI()
```

#### `get_response(message: str, session_id: str = None) -> Dict`
Send a message and get Duck's response.

**Parameters:**
- `message` (str): User's message
- `session_id` (str, optional): Chat session ID (creates new if not provided)

**Returns:**
```json
{
  "status": "success",
  "session_id": "session_1",
  "user_message": "Hello Duck",
  "duck_response": "*beep boop* Hey there! ...",
  "message_count": 2,
  "personality": {
    "humor": "R2-D2",
    "versatility": "C-3PO"
  }
}
```

**Example:**
```python
response = duck.get_response("Tell me a joke")
print(response['duck_response'])
```

#### `create_session(session_id: str = None) -> str`
Create a new chat session.

**Parameters:**
- `session_id` (str, optional): Custom session ID

**Returns:** Session ID (str)

**Example:**
```python
session_id = duck.create_session("my_chat")
```

#### `get_session(session_id: str) -> Dict`
Retrieve a chat session's history.

**Parameters:**
- `session_id` (str): Session ID

**Returns:**
```json
{
  "status": "success",
  "session": {
    "id": "session_1",
    "messages": [...],
    "created_at": "2025-12-02T10:30:45.123456",
    "status": "active"
  }
}
```

**Example:**
```python
session = duck.get_session("session_1")
for msg in session['session']['messages']:
    print(f"{msg['role']}: {msg['content']}")
```

#### `list_sessions() -> Dict`
List all active chat sessions.

**Returns:**
```json
{
  "status": "success",
  "total_sessions": 3,
  "sessions": [
    {
      "id": "session_1",
      "created_at": "2025-12-02T10:30:45",
      "message_count": 5,
      "status": "active"
    }
  ]
}
```

#### `get_personality() -> Dict`
Get Duck's personality configuration.

**Returns:**
```json
{
  "status": "success",
  "personality": {
    "model_name": "Duck (Super APL Model)",
    "personality_profile": {
      "humor_style": "R2-D2",
      "versatility_style": "C-3PO",
      "system_prompt": "You are Duck..."
    },
    "training_parameters": {
      "quantization": "NF4",
      "context_window": 4096,
      "epochs": 3
    }
  }
}
```

#### `clear_session(session_id: str) -> Dict`
Clear a chat session.

**Parameters:**
- `session_id` (str): Session ID to clear

**Returns:**
```json
{
  "status": "success",
  "message": "Session session_1 cleared"
}
```

## REST API Endpoints

### Base URL
```
http://localhost:5000/api/v1
```

### 1. Send Message to Duck
**Endpoint:** `POST /chat`

**Request:**
```json
{
  "message": "Hello Duck, tell me about yourself",
  "session_id": "optional_session_id"
}
```

**Response:**
```json
{
  "status": "success",
  "session_id": "session_1",
  "user_message": "Hello Duck, tell me about yourself",
  "duck_response": "*proud beeps* I'm Duck! ...",
  "message_count": 1,
  "personality": {
    "humor": "R2-D2",
    "versatility": "C-3PO"
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me a joke",
    "session_id": "my_session"
  }'
```

### 2. Create New Session
**Endpoint:** `POST /session`

**Request:**
```json
{
  "session_id": "optional_custom_id"
}
```

**Response:**
```json
{
  "status": "success",
  "session_id": "session_1"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/v1/session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "chat_with_duck"}'
```

### 3. Get Session Details
**Endpoint:** `GET /session/<session_id>`

**Response:**
```json
{
  "status": "success",
  "session": {
    "id": "session_1",
    "messages": [
      {
        "role": "user",
        "content": "Hello Duck",
        "timestamp": "2025-12-02T10:30:45"
      },
      {
        "role": "duck",
        "content": "*beep boop* ...",
        "timestamp": "2025-12-02T10:30:46"
      }
    ],
    "created_at": "2025-12-02T10:30:44",
    "status": "active"
  }
}
```

**Example:**
```bash
curl http://localhost:5000/api/v1/session/session_1
```

### 4. List All Sessions
**Endpoint:** `GET /sessions`

**Response:**
```json
{
  "status": "success",
  "total_sessions": 2,
  "sessions": [
    {
      "id": "session_1",
      "created_at": "2025-12-02T10:30:45",
      "message_count": 5,
      "status": "active"
    },
    {
      "id": "session_2",
      "created_at": "2025-12-02T10:35:20",
      "message_count": 2,
      "status": "active"
    }
  ]
}
```

**Example:**
```bash
curl http://localhost:5000/api/v1/sessions
```

### 5. Clear Session
**Endpoint:** `DELETE /session/<session_id>`

**Response:**
```json
{
  "status": "success",
  "message": "Session session_1 cleared"
}
```

**Example:**
```bash
curl -X DELETE http://localhost:5000/api/v1/session/session_1
```

### 6. Get Duck's Personality
**Endpoint:** `GET /personality`

**Response:**
```json
{
  "status": "success",
  "personality": {
    "model_name": "Duck (Super APL Model)",
    "personality_profile": {
      "humor_style": "R2-D2",
      "humor_description": "Sassy, expressive, beep-boop sarcasm, brave, cheeky",
      "versatility_style": "C-3PO",
      "versatility_description": "Fluent, protocol-focused, helpful",
      "system_prompt": "You are Duck..."
    }
  }
}
```

### 7. Get API Status
**Endpoint:** `GET /status`

**Response:**
```json
{
  "status": "online",
  "service": "Duck Chat API",
  "version": "1.0.0",
  "personality": "R2-D2 + C-3PO",
  "training": "Llama APL",
  "active_sessions": 2
}
```

## CLI Commands

### Chat Command
Send a message to Duck and get a response.

```bash
python src/api/duck_chat_api.py chat <message> [--session SESSION_ID]
```

**Examples:**
```bash
python src/api/duck_chat_api.py chat "Hello Duck"
python src/api/duck_chat_api.py chat "Help with matrices" --session chat_1
```

### Server Command
Start the REST API server.

```bash
python src/api/duck_chat_api.py server [--host HOST] [--port PORT]
```

**Options:**
- `--host` (default: localhost) - Host to bind to
- `--port` (default: 5000) - Port number

**Examples:**
```bash
python src/api/duck_chat_api.py server
python src/api/duck_chat_api.py server --port 8080
python src/api/duck_chat_api.py server --host 0.0.0.0 --port 5000
```

### Status Command
Check API status.

```bash
python src/api/duck_chat_api.py status
```

### Personality Command
Show Duck's personality configuration.

```bash
python src/api/duck_chat_api.py personality
```

### Sessions Command
List active chat sessions.

```bash
python src/api/duck_chat_api.py sessions
```

## Topics Duck Understands

### Greetings
- "hello", "hi", "hey" → Friendly greeting with Duck personality

### Jokes & Humor
- "joke", "funny", "laugh" → Gets a joke with R2-D2 attitude

### APL/Matrix Questions
- "matrix", "apl", "compute", "multiply", "transpose" → Explains with Llama APL training

### Self Description
- "who are you", "what are you", "tell me about you" → Full personality introduction

### Status/Diagnostics
- "status", "check", "diagnostic" → System status report

### General Conversation
- Any other input → Intelligent response with personality

## Session Management

### How Sessions Work
1. **Automatic**: First message creates a session automatically
2. **Manual**: Create with `create_session()` or POST `/session`
3. **Persistent**: Messages stored in session throughout conversation
4. **Reusable**: Reference same `session_id` in multiple requests

### Best Practices
```python
# Create session once
session_id = duck.create_session("my_chat")

# Reuse in multiple messages
response1 = duck.get_response("Hello", session_id)
response2 = duck.get_response("How are you?", session_id)
response3 = duck.get_response("Tell me a joke", session_id)

# View full conversation
session = duck.get_session(session_id)
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "status": "error",
  "message": "Error description"
}
```

**Common Errors:**

| Error | Meaning | Solution |
|-------|---------|----------|
| "Message required" | Empty message sent | Provide a non-empty message |
| "Session not found" | Invalid session ID | Create new session or check ID |
| "Flask not installed" | REST API unavailable | `pip install flask` |

## Performance & Limits

| Metric | Value |
|--------|-------|
| Response Time | < 100ms |
| Max Sessions | Unlimited (memory limited) |
| Max Message Length | 10,000 characters |
| Context Window | 4096 tokens |
| Quantization | NF4 4-bit |

## Examples

### Multi-turn Conversation (Python)
```python
from src.api.duck_chat_api import DuckChatAPI

duck = DuckChatAPI()
session = duck.create_session()

# Multi-turn conversation
messages = [
    "Hello Duck",
    "What can you help with?",
    "Tell me a joke",
    "Thanks, that was funny!"
]

for msg in messages:
    response = duck.get_response(msg, session)
    print(f"You: {msg}")
    print(f"Duck: {response['duck_response']}\n")
```

### REST API with curl
```bash
# Create session
SESSION=$(curl -s -X POST http://localhost:5000/api/v1/session | jq -r '.session_id')

# Send messages
curl -X POST http://localhost:5000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Hello Duck\", \"session_id\": \"$SESSION\"}"

# Get conversation history
curl http://localhost:5000/api/v1/session/$SESSION | jq '.session.messages'
```

### REST API with Python requests
```python
import requests

BASE_URL = "http://localhost:5000/api/v1"

# Create session
session_resp = requests.post(f"{BASE_URL}/session")
session_id = session_resp.json()['session_id']

# Send messages
for message in ["Hello Duck", "Tell me a joke", "Thanks!"]:
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": message, "session_id": session_id}
    )
    result = response.json()
    print(f"Duck: {result['duck_response']}")

# Get full conversation
session = requests.get(f"{BASE_URL}/session/{session_id}")
print(f"Messages: {len(session.json()['session']['messages'])}")
```

## Deployment

### Standalone Executable (Windows)
Pre-built executable available in `dist/DuckChat.exe` (11.03 MB)

```bash
# Start REST API server with default settings (localhost:5000)
dist/DuckChat.exe

# Or with custom host/port
dist/DuckChat.exe server --host 0.0.0.0 --port 8080
```

No Python installation required—just run the exe!

### Development Server
```bash
python src/api/duck_chat_api.py server --host localhost --port 5000
```

### Production Server (with gunicorn)
```bash
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 src.api.duck_chat_api:create_flask_app
```

### Docker
```dockerfile
FROM python:3.12
WORKDIR /app
COPY . .
RUN pip install flask
CMD ["python", "src/api/duck_chat_api.py", "server", "--host", "0.0.0.0"]
```

## Status
✅ Production Ready  
✅ All endpoints tested  
✅ Error handling implemented  
✅ Multiple access methods  
✅ Session persistence  
✅ Full personality integration
