Duck RAG integration
--------------------

This project adds a Chroma (chromadb) based RAG index for the `library/` folder.

Requirements:

- Python packages: chromadb, sentence-transformers
- Optional: faiss-cpu for faster searches

Install:

```powershell
pip install chromadb sentence-transformers
# Optional: pip install chromadb[faiss]
# or conda install -c conda-forge faiss-cpu
```

Basic usage:

Run a one-off reindex (from the repository root):

```powershell
python -c "from src.api.duck_chat_api import DuckChatAPI; DuckChatAPI()._build_vector_store_from_library()"
```

Start server from source:

```powershell
python src/api/duck_chat_api.py server --host localhost --port 5000
```

Use REST to reindex and query:

```powershell
# Reindex
curl -X POST http://localhost:5000/api/v1/library/reindex

# Chat
curl -X POST -H "Content-Type: application/json" -d '{"message":"What does the library say about APL arrays?"}' http://localhost:5000/api/v1/chat

# Load adapter
curl -X POST -H "Content-Type: application/json" -d '{"adapter":"mylora"}' http://localhost:5000/api/v1/adapter/load
```

Notes:
- If chromadb or sentence-transformers are missing, Duck will gracefully skip the index/retrieval and fallback to reading raw `library` files as context.
- If chromadb is installed and the `dev` DB is used, you can inspect the data in `db/chromadb_library`.

LoRA training & synthetic QA generation
-------------------------------------

1) Generate synthetic QA pairs (requires the server to be running so Duck can generate answers):

```powershell
python src/training/generate_qas.py
```

This will produce `training/library_dataset.jsonl` containing instruction+output pairs.

2) Train LoRA adapter from the synthetic dataset:

```powershell
pip install transformers datasets accelerate bitsandbytes peft
python src/training/lora_train.py --model mistralai/Mistral-7B-Instruct-v0.2 --dataset training/library_dataset.jsonl --output adapters/mylora
```

3) Load the adapter at runtime by modifying `duck_chat_api.py` to load the adapter weights into the model. (This is not a change applied automatically.)

Be cautious: LoRA training requires hardware resources and may require `accelerate` configuration for proper multi-GPU or single GPU usage.
