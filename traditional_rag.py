import os
import textwrap
from typing import List, Dict, Any, Tuple
from collections import deque

from dotenv import load_dotenv

from pinecone_db import PineconeDB
from llm import GroqLLM


def retrieve_from_both_indexes(query: str, top_k_each: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks from both indexes/namespaces and return merged matches.

    Returns list of items with: id, score, metadata, source (index/namespace).
    """
    sources: List[Tuple[str, str]] = [
        ("langtech-langchain", "langchain-data"),
        ("langtech-langgraph", "langgraph-data"),
    ]

    merged: List[Dict[str, Any]] = []
    for index_name, namespace in sources:
        db = PineconeDB(index_name=index_name)
        db.set_namespace(namespace)

        vec = db.get_embedding(query)
        res = db.index.query(
            vector=vec,
            top_k=top_k_each,
            include_values=False,
            include_metadata=True,
            namespace=namespace,
        )
        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        for m in matches:
            if isinstance(m, dict):
                merged.append({
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "metadata": m.get("metadata") or {},
                    "source": {"index": index_name, "namespace": namespace},
                })
            else:
                merged.append({
                    "id": getattr(m, "id", None),
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", {}) or {},
                    "source": {"index": index_name, "namespace": namespace},
                })

    # Sort merged by score descending
    merged.sort(key=lambda x: x.get("score") or 0.0, reverse=True)
    return merged


def build_context(matches: List[Dict[str, Any]], max_chunks: int = 4) -> str:
    parts = []
    for m in matches[:max_chunks]:
        meta = m.get("metadata") or {}
        title = meta.get("title", "")
        txt = meta.get("chunk_text", "")
        src = m.get("source", {})
        parts.append(
            f"[source: {src.get('index')}/{src.get('namespace')} | id: {m.get('id')} | score: {m.get('score'):.4f} | title: {title}]\n{txt}"
        )
    return "\n\n".join(parts)


def format_history(history: List[Tuple[str, str]], max_turns: int = 4) -> str:
    if not history:
        return ""
    recent = list(history)[-max_turns:]
    lines = []
    for u, a in recent:
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def answer_with_rag(query: str, matches: List[Dict[str, Any]], history: List[Tuple[str, str]]) -> str:
    context = build_context(matches)
    convo = format_history(history, max_turns=4)
    prompt = f"""
You are a helpful assistant in a short, ongoing conversation. Use the Context to answer. Respect the Conversation so far when resolving references (e.g., follow-ups). If the answer isn't in the Context, say you don't know.

Conversation:
{convo}

Context:
{context}

Question: {query}
Answer in 3-5 concise sentences.
""".strip()
    llm = GroqLLM()
    # Use invoke to avoid deprecation warnings
    return llm.invoke(prompt)


def cli():
    load_dotenv()
    print("RAG CLI across two indexes (langtech-langchain, langtech-langgraph). Type 'exit' to quit.\n")
    # Ephemeral, in-memory history (not persisted to disk)
    history: deque[Tuple[str, str]] = deque(maxlen=8)  # keep last ~4 turns (8 messages)
    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q", "q"}:
            break

        matches = retrieve_from_both_indexes(q, top_k_each=3)
        if not matches:
            print("Assistant: I couldn't find relevant context.")
            continue

        answer = answer_with_rag(q, matches, history)
        print("Assistant:")
        print(textwrap.fill(answer, width=100))
        history.append((q, answer))


if __name__ == "__main__":
    cli()
