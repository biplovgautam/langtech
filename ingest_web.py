import argparse
import hashlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
from dotenv import load_dotenv

from pinecone_db import PineconeDB


LANGCHAIN_START = "https://python.langchain.com/docs/"
LANGGRAPH_START = "https://langchain-ai.github.io/langgraph/"


@dataclass
class CrawlTarget:
    name: str
    start_urls: List[str]
    allow_prefixes: List[str]
    index_name: str
    namespace: str


def normalize_url(url: str) -> str:
    clean, _ = urldefrag(url)
    return clean.rstrip("/")


def in_scope(url: str, prefixes: List[str]) -> bool:
    url = url.lower()
    for p in prefixes:
        if url.startswith(p.lower()):
            return True
    return False


def fetch(url: str, timeout: int = 20) -> Optional[str]:
    try:
        headers = {
            "User-Agent": "langtech-ingest/1.0 (+https://github.com/biplovgautam/langtech)",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")
        if "text/html" not in ct:
            return None
        return r.text
    except requests.RequestException:
        return None


def extract_content(html: str) -> Tuple[str, str]:
    """Return (title, text) extracted from HTML."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()
    body = soup.get_text("\n", strip=True)
    # Normalize whitespace
    lines = [ln.strip() for ln in body.splitlines()]
    text = "\n".join([ln for ln in lines if ln])
    return title, text


def find_links(html: str, base_url: str, allow_prefixes: List[str]) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = normalize_url(abs_url)
        if in_scope(abs_url, allow_prefixes):
            links.append(abs_url)
    return links


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def records_from_page(url: str, title: str, text: str) -> List[Dict[str, Any]]:
    chunks = chunk_text(text)
    recs: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        uid = hashlib.md5(f"{url}#{i}".encode("utf-8")).hexdigest()
        recs.append({
            "_id": uid,
            "chunk_text": chunk,
            "url": url,
            "title": title,
            "category": "docs",
            "chunk_index": i,
        })
    return recs


def crawl_and_ingest(target: CrawlTarget, max_pages: int, batch_size: int = 40) -> Tuple[int, int]:
    db = PineconeDB(index_name=target.index_name)
    db.set_namespace(target.namespace)

    visited: Set[str] = set()
    seeds = [normalize_url(u) for u in target.start_urls]
    q: deque[str] = deque(seeds)
    pages_seen = 0
    total_chunks = 0
    batch: List[Dict[str, Any]] = []

    while q and pages_seen < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        html = fetch(url)
        if not html:
            continue

        title, text = extract_content(html)
        page_recs = records_from_page(url, title, text)
        batch.extend(page_recs)
        total_chunks += len(page_recs)
        pages_seen += 1

        # enqueue new links
        for link in find_links(html, url, target.allow_prefixes):
            if link not in visited:
                q.append(link)

        # upsert in batches
        if len(batch) >= batch_size:
            db.append_records(batch, namespace=target.namespace)
            batch.clear()
            # Be gentle with inference API rate limits
            time.sleep(0.5)

    # flush remaining
    if batch:
        db.append_records(batch, namespace=target.namespace)

    return pages_seen, total_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest LangChain and LangGraph docs into Pinecone indexes.")
    p.add_argument("--target", choices=["langchain", "langgraph", "both"], default="both")
    p.add_argument("--langchain-start", action="append", help="Start URL(s) for LangChain docs (can repeat)")
    p.add_argument("--langgraph-start", action="append", help="Start URL(s) for LangGraph docs (can repeat)")
    p.add_argument("--langchain-index", default="langtech-langchain")
    p.add_argument("--langgraph-index", default="langtech-langgraph")
    p.add_argument("--langchain-ns", default="langchain-data")
    p.add_argument("--langgraph-ns", default="langgraph-data")
    p.add_argument("--max-pages", type=int, default=50, help="Max pages to crawl per target")
    p.add_argument("--batch-size", type=int, default=40, help="Upsert batch size (chunks)")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Defaults for seeds and allowed prefixes
    default_langchain_starts = [
        LANGCHAIN_START,
        "https://python.langchain.com/v0.2/docs/",
        "https://api.python.langchain.com/en/latest/",
    ]
    default_langgraph_starts = [
        LANGGRAPH_START,
    ]

    lc_starts = [normalize_url(u) for u in (args.langchain_start or default_langchain_starts)]
    lg_starts = [normalize_url(u) for u in (args.langgraph_start or default_langgraph_starts)]

    targets: List[CrawlTarget] = []
    if args.target in ("langchain", "both"):
        targets.append(CrawlTarget(
            name="langchain",
            start_urls=lc_starts,
            allow_prefixes=lc_starts,
            index_name=args.langchain_index,
            namespace=args.langchain_ns,
        ))
    if args.target in ("langgraph", "both"):
        targets.append(CrawlTarget(
            name="langgraph",
            start_urls=lg_starts,
            allow_prefixes=lg_starts,
            index_name=args.langgraph_index,
            namespace=args.langgraph_ns,
        ))

    for t in targets:
        print(f"\n>>> Ingesting {t.name} docs from {len(t.start_urls)} seed(s) into index={t.index_name} ns={t.namespace}")
        pages, chunks = crawl_and_ingest(t, max_pages=args.max_pages, batch_size=args.batch_size)
        print(f"Done {t.name}: pages={pages}, chunks={chunks}")


if __name__ == "__main__":
    main()
