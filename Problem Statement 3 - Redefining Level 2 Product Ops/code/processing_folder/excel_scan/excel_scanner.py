# excel_scanner.py (read-only; never builds cache)
import os, pickle, argparse, json
from dataclasses import dataclass
from typing import List, Tuple, Optional

from embedding_helper import get_embedding, get_cosine_similarity

# ---- CONFIG ---------------------------------------------------------------
DEFAULT_XLSX = "/Users/zian/Documents/PSA Hackathon/PSA_Mediocre/Problem Statement 3 - Redefining Level 2 Product Ops/Info/Case Log.xlsx"
DEFAULT_CACHE = "case_log.embcache.pkl"   # must already exist

# ---- META TYPES -----------------------------------------------------------
@dataclass
class CacheMeta:
    xlsx_path: str
    file_sha256: str
    file_size: int
    file_mtime: float
    model_id: str
    code_version: str
    created_at: float
    rows_limited_to: Optional[int]

@dataclass
class CacheItem:
    sheet: str
    row: int
    col: int
    text: str
    emb: List[float]

@dataclass
class ExcelCache:
    meta: CacheMeta
    items: List[CacheItem]

# ---- LOAD-ONLY ------------------------------------------------------------
def _load_cache(cache_path: str) -> Optional[ExcelCache]:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "meta" in data and "items" in data:
        meta = data["meta"]
        if isinstance(meta, dict):
            meta = CacheMeta(**meta)
        items: List[CacheItem] = [
            CacheItem(**it) if isinstance(it, dict) else it for it in data["items"]
        ]
        return ExcelCache(meta=meta, items=items)
    return data

# ---- SEARCH (CACHED) ------------------------------------------------------
def semantic_excel_search_cached(
    query: str,
    cache_path: str = DEFAULT_CACHE,
    threshold: float = 0.7,
    top_k: Optional[int] = None
) -> List[Tuple[float, str, str, int, int]]:
    """
    Compare the query embedding with cached cell embeddings.
    Returns: (similarity, text, sheet, row, col) sorted desc by similarity.
    """
    cache = _load_cache(cache_path)
    if cache is None:
        raise FileNotFoundError(
            f"Cache not found at '{cache_path}'. Build it once with your caching script."
        )

    q_emb = get_embedding(query)
    if not q_emb:
        return []

    out: List[Tuple[float, str, str, int, int]] = []
    for it in cache.items:
        sim = get_cosine_similarity(q_emb, it.emb)
        if sim >= threshold:
            out.append((sim, it.text, it.sheet, it.row, it.col))

    out.sort(key=lambda t: t[0], reverse=True)
    return out[:top_k] if top_k else out

def check_in_excel_cached(
    query: str,
    threshold: float = 0.7,
    top_k: int = 5,
    cache_path: str = DEFAULT_CACHE
):
    hits = semantic_excel_search_cached(
        query, cache_path=cache_path, threshold=threshold, top_k=top_k
    )
    return (len(hits) > 0, hits)

def print_hits(hits):
    if not hits:
        print("❌ No semantic matches at or above the threshold.")
        return
    for sim, text, sheet, r, c in hits:
        print(f"{sim:.3f} | [{sheet} r{r} c{c}] {text}")

# ---- CLI (SEARCH ONLY) ----------------------------------------------------
def _cli():
    p = argparse.ArgumentParser(description="Search Excel cache (read-only).")
    p.add_argument("--cache", type=str, default=DEFAULT_CACHE, help="Path to .embcache.pkl")
    p.add_argument("--query", type=str, required=True, help="Query text to search")
    p.add_argument("--th", type=float, default=0.7, help="Similarity threshold")
    p.add_argument("--topk", type=int, default=10, help="Top-K results")
    return p.parse_args()

def check_excel_for_string(query: str, 
                           threshold: float = 0.7, 
                           top_k: int = 10, 
                           cache_path: str = DEFAULT_CACHE):
    """
    Check if a given string (query) appears semantically in the Excel cache.

    Args:
        query (str): The text to search for.
        threshold (float): Similarity threshold (0–1). Default = 0.7.
        top_k (int): Number of top matches to return.
        cache_path (str): Path to the existing .embcache.pkl file.

    Returns:
        (bool, List[Tuple[float, str, str, int, int]]):
        - bool: True if any match ≥ threshold
        - List of tuples: (similarity, text, sheet, row, col)
    """
    cache_exists = os.path.exists(cache_path)
    if not cache_exists:
        raise FileNotFoundError(f"❌ Cache not found: {cache_path}")

    # Perform the semantic check
    present, hits = check_in_excel_cached(query, threshold=threshold, top_k=top_k, cache_path=cache_path)
    
    # Optional pretty print
    print(f"\n🔍 Checking for: '{query}'")
    print("✅ Found matches:\n" if present else "❌ No matches found.\n")
    print_hits(hits)

    return present, hits


if __name__ == "__main__":
    """check_excel_for_string(
        "Notification: SMS TCK-936729 — Detected an ANSI X12 301 data mismatch for vessel MV SILVER CURRENT/43C. The COARRI message indicates discharge finished at bay 22, b. Kindly verify urgently.",
        threshold=0.7
    )"""


    string_data = json.dumps("EMAIL ALR-861600 | CMAU00000020 - DUPLICATE CONTAINER INFORMATION RECEIVED  HI JEN  PLEASE ASSIST IN CHECKING CONTAINER CMAU00000020  CUSTOMER ON PORTNET IS SEEING 2 IDENTICAL CONTAINERS INFORMATION ")
    print(string_data)


    check_excel_for_string(string_data, threshold=0.7)
