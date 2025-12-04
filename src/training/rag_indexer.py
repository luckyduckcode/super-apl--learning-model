"""
RAG Indexer: build and query a chromadb-based vector store from `library/` files.

This module: 
- builds a Chroma collection from text files in `library/` (chunked into 1500-character blocks)
- can be queried to return top-k matched chunks with their metadata

It gracefully handles the absence of Chroma or sentence-transformers by returning False or empty results.
"""

import os
import math

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    S2_AVAILABLE = True
except Exception:
    S2_AVAILABLE = False


def _get_default_persist_dir():
    return os.path.abspath(os.path.join(os.getcwd(), "db", "chromadb_library"))


def build_index_from_library(base_path=None, persist_directory=None, chunk_size=1500):
    """Scan `base_path/library` and index all .txt/.md files into Chroma.

    Returns True on success, False on failure or if no documents found or chromadb missing.
    """
    if not CHROMADB_AVAILABLE:
        print("[RAG] chromadb not available; skipping build_index_from_library")
        return False

    if not S2_AVAILABLE:
        print("[RAG] sentence-transformers not available; skipping build_index_from_library")
        return False

    if base_path is None:
        base_path = os.getcwd()

    lib_path = os.path.join(base_path, "library")
    if not os.path.exists(lib_path):
        print(f"[RAG] Library path doesn't exist: {lib_path}")
        return False

    if persist_directory is None:
        persist_directory = _get_default_persist_dir()

    os.makedirs(persist_directory, exist_ok=True)

    try:
        # Initialize chroma client
        # Use Settings with only persist_directory to support different chromadb versions
        try:
            client = chromadb.Client(Settings(persist_directory=persist_directory))
        except Exception:
            # Fallback: try passing persist_directory directly
            client = chromadb.Client(persist_directory=persist_directory)
        coll_name = "duck_library"
        # delete and recreate to ensure a fresh index each run
        existing_collections = [c.name for c in client.list_collections()]
        if coll_name in existing_collections:
            try:
                coll = client.get_collection(coll_name)
                # Recreate collection: delete and create a fresh one
                client.delete_collection(coll_name)
            except Exception:
                pass

        try:
            coll = client.create_collection(name=coll_name)
        except Exception:
            # older versions might require different args
            coll = client.create_collection(coll_name)

        # Create embedding model; chroma supports built-in embeddings but we'll use a simple SBERT embedder
        model = SentenceTransformer("all-MiniLM-L6-v2")

        documents = []
        metadatas = []
        ids = []
        idx = 0

        for filename in sorted(os.listdir(lib_path)):
            if not (filename.lower().endswith('.txt') or filename.lower().endswith('.md')):
                continue
            path = os.path.join(lib_path, filename)
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    text = fh.read().strip()
            except Exception as e:
                print(f"[RAG] Could not read {path}: {e}")
                continue

            if not text:
                continue

            # chunk text
            start = 0
            L = len(text)
            while start < L:
                chunk = text[start:start+chunk_size]
                # simple heuristic to avoid empty or tiny chunks
                if len(chunk.strip()) < 20:
                    break
                documents.append(chunk)
                metadatas.append({"source": filename, "chunk_index": idx})
                ids.append(f"lib_{idx}")
                idx += 1
                start += chunk_size

        if not documents:
            print("[RAG] No documents found in library to index.")
            return False

        # Add to collection (Chroma computes embeddings when not provided)
        coll.add(documents=documents, metadatas=metadatas, ids=ids)
        # Persist the client if supported
        try:
            if hasattr(client, 'persist'):
                client.persist()
        except Exception:
            try:
                if hasattr(client, 'persist'):
                    client.persist()
                elif hasattr(client, 'store'):
                    client.store()
            except Exception:
                # Not all versions support persist/store; ignore
                pass
        print(f"[RAG] Indexed {len(documents)} chunks from {lib_path} into {persist_directory}")
        return True

    except Exception as e:
        print(f"[RAG] Failed to build index: {e}")
        return False


def query_index(query_text: str, persist_directory=None, k=3):
    """Query Chroma for the top-k chunks related to query_text and return as a list of dicts.

    Each result dict contains: {'document': text, 'source': filename, 'chunk_index': int}
    Returns [] if no results or chromadb missing.
    """
    if not CHROMADB_AVAILABLE:
        return []

    if persist_directory is None:
        persist_directory = _get_default_persist_dir()

    try:
        try:
            client = chromadb.Client(Settings(persist_directory=persist_directory))
        except Exception:
            client = chromadb.Client(persist_directory=persist_directory)
        coll_name = "duck_library"
        existing_collections = [c.name for c in client.list_collections()]
        if coll_name not in existing_collections:
            return []
        coll = client.get_collection(coll_name)
        results = coll.query(query_texts=[query_text], n_results=k, include=["documents", "metadatas"])  # returns a map
        docs = results.get("documents", [[]])[0] if "documents" in results else []
        metas = results.get("metadatas", [[]])[0] if "metadatas" in results else []
        out = []
        for md, doc in zip(metas, docs):
            out.append({"document": doc, "source": md.get('source'), "chunk_index": md.get('chunk_index')})
        return out
    except Exception as e:
        print(f"[RAG] Query failed: {e}")
        return []
