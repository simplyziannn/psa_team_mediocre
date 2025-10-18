import os, sys, json, hashlib, pickle, re
from typing import List, Tuple, Iterable, Dict, Optional, Any
from pathlib import Path
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table

# your helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from excel_scan.embedding_helper import get_embedding, get_cosine_similarity


# ---------------------------- Config ---------------------------------

DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K     = 10
MAX_CHARS_PER_CHUNK = 500
EMB_CACHE_DIR = Path(__file__).parent / ".emb_cache"

# ---------------------------------------------------------------------


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _chunk_text(s: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    s = _normalize_text(s)
    if len(s) <= max_chars:
        return [s] if s else []
    chunks, start = [], 0
    while start < len(s):
        end = min(start + max_chars, len(s))
        cut = s.rfind(". ", start, end)
        if cut == -1 or cut < start + int(max_chars * 0.5):
            cut = end
        else:
            cut += 2
        chunks.append(s[start:cut].strip())
        start = cut
    return [c for c in chunks if c]


def _iter_docx_texts(docx_path: str, include_tables: bool = True,
                     chunk: bool = True, max_chars: int = MAX_CHARS_PER_CHUNK
                     ) -> Iterable[Tuple[str, str]]:
    """
    Yields (source_id, text) from the .docx file:
      - Paragraphs as P#<idx>
      - (Optional) Table cells as T<table_idx>R<row_idx>C<col_idx>
    """
    doc = Document(docx_path)

    # Paragraphs
    for i, p in enumerate(doc.paragraphs):
        t = _normalize_text(p.text or "")
        if not t:
            continue
        if chunk:
            for j, ch in enumerate(_chunk_text(t, max_chars=max_chars)):
                yield (f"P#{i}:{j}", ch)
        else:
            yield (f"P#{i}", t)

    # Tables
    if include_tables:
        for ti, tbl in enumerate(doc.tables):
            for ri, row in enumerate(tbl.rows):
                for ci, cell in enumerate(row.cells):
                    t = _normalize_text(cell.text or "")
                    if not t:
                        continue
                    if chunk:
                        for j, ch in enumerate(_chunk_text(t, max_chars=max_chars)):
                            yield (f"T{ti}R{ri}C{ci}:{j}", ch)
                    else:
                        yield (f"T{ti}R{ri}C{ci}", t)


# ------------------------ Embedding cache -----------------------------

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _load_cache() -> Dict[str, List[float]]:
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pkl = EMB_CACHE_DIR / "embeddings.pkl"
    if pkl.exists():
        try:
            with open(pkl, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache: Dict[str, List[float]]) -> None:
    pkl = EMB_CACHE_DIR / "embeddings.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(cache, f)

def _embed_with_cache(text: str, cache: Dict[str, List[float]]) -> Optional[List[float]]:
    key = _sha1(text)
    if key in cache:
        return cache[key]
    emb = get_embedding(text)
    if emb:
        cache[key] = emb
    return emb


# ------------------------ Main search API -----------------------------

def semantic_docx_search(
    query: str,
    docx_path: str,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    include_tables: bool = True,
    chunk: bool = True,
    max_chars: int = MAX_CHARS_PER_CHUNK,
    return_always_topk: bool = True,
    keyword_fallback: bool = True,
) -> List[Tuple[str, str, float]]:
    """
    Semantic search over .docx.
    Returns up to top_k results sorted by similarity desc.
    Tuple fields: (source_id, matched_text, similarity_score)
    """
    if not os.path.exists(docx_path):
        return []

    cache = _load_cache()

    query_norm = _normalize_text(query)
    query_emb = get_embedding(query_norm)
    if not query_emb:
        return []

    matches: List[Tuple[str, str, float]] = []
    seen_text: Dict[str, float] = {}

    for source_id, text in _iter_docx_texts(
        docx_path, include_tables=include_tables, chunk=chunk, max_chars=max_chars
    ):
        if not text:
            continue
        if text in seen_text:
            sim = seen_text[text]
        else:
            emb = _embed_with_cache(text, cache)
            if not emb:
                continue
            sim = get_cosine_similarity(query_emb, emb)
            seen_text[text] = sim
        matches.append((source_id, text, sim))

    _save_cache(cache)

    matches.sort(key=lambda x: x[2], reverse=True)
    filtered = [m for m in matches if m[2] >= threshold]
    if filtered:
        return filtered[:top_k] if top_k else filtered

    if return_always_topk and matches:
        return matches[:top_k] if top_k else matches

    if keyword_fallback:
        q_words = set(w.lower() for w in query_norm.split() if len(w) > 3)
        kmatches = []
        for sid, text, _sim in matches:
            tset = set(text.lower().split())
            overlap = len(q_words & tset)
            if overlap >= 2:
                kmatches.append((sid, text, 0.0))
        if kmatches:
            return kmatches[:top_k] if top_k else kmatches

    return []


# ------------------------ Headings / doc structure --------------------

def _heading_level(p: Paragraph) -> Optional[int]:
    """Returns heading level (1..9) if paragraph style is 'Heading N'; else None."""
    try:
        name = (p.style.name or "").strip()
    except Exception:
        name = ""
    m = re.match(r"Heading\s+([1-9])$", name, flags=re.I)
    return int(m.group(1)) if m else None

def _is_list_item(p: Paragraph) -> bool:
    """Heuristic: Word numbering (numPr) OR a visible bullet/number prefix."""
    try:
        numPr = p._p.pPr.numPr  # type: ignore[attr-defined]
        if numPr is not None:
            return True
    except Exception:
        pass
    t = (p.text or "").strip()
    return bool(re.match(r"^([\-–•●◦·]|(\d+|[A-Za-z])[\.\)]|•)\s+", t))

def iter_block_items(doc: Document):
    """Yields blocks in document order as ('p', Paragraph) or ('tbl', Table)."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag.endswith('p'):
            yield ('p', Paragraph(child, doc))
        elif child.tag.endswith('tbl'):
            yield ('tbl', Table(child, doc))

def build_doc_index(doc: Document) -> Dict[str, int]:
    """Map 'P#<i>' and 'T<i>' to block index."""
    idx: Dict[str, int] = {}
    p_counter, t_counter, b_counter = 0, 0, 0
    for kind, _obj in iter_block_items(doc):
        if kind == 'p':
            idx[f"P#{p_counter}"] = b_counter
            p_counter += 1
        else:
            idx[f"T{t_counter}"] = b_counter
            t_counter += 1
        b_counter += 1
    return idx

def _parse_source_id_root(source_id: str) -> Optional[str]:
    """Normalize source_id to paragraph/table root id used in build_doc_index."""
    m = re.match(r"^(P#\d+)(?::\d+)?$", source_id)
    if m: return m.group(1)
    m = re.match(r"^(T\d+)R\d+C\d+(?::\d+)?$", source_id)
    if m: return m.group(1)
    m = re.match(r"^(T\d+)(?::\d+)?$", source_id)
    if m: return m.group(1)
    return None


# -------- Owner detection, classification, and module extraction ------

_OWNER_PREFIX_RE = re.compile(r"^\s*([A-Z]{3,5})\s*:\s*", re.I)

def _classify_owner(title: str) -> Dict[str, Optional[str]]:
    """Return owner 'prefix' (CNTR/VSL/API/EDI/...) and the 'family' label."""
    title = (title or "").strip()
    m = _OWNER_PREFIX_RE.match(title)
    prefix = (m.group(1).upper() if m else None)
    # family buckets (you can extend)
    if prefix in {"CNTR"}:
        family = "CNTR"
    elif prefix in {"VSL", "VSSL", "VESSEL"}:
        family = "VSL"
    elif prefix in {"API"}:
        family = "API"
    elif prefix in {"EDI"}:
        family = "EDI"
    else:
        family = prefix or None
    return {"prefix": prefix, "family": family}

def _find_owner_bounds(blocks: List[Tuple[str, Any]], hit_block: int, max_scan: int = 800) -> Tuple[int, int, Optional[int]]:
    """Find nearest heading above hit_block and its end (next heading <= its level)."""
    start, lvl = None, None
    # find heading up
    for i in range(hit_block, max(-1, hit_block - max_scan), -1):
        kind, obj = blocks[i]
        if kind == 'p':
            hl = _heading_level(obj)  # type: ignore[arg-type]
            if hl is not None:
                start, lvl = i, hl
                break
    if start is None:
        start, lvl = hit_block, 10
    # find end
    end = len(blocks) - 1
    for k in range(start + 1, min(len(blocks), start + 1 + max_scan)):
        kind, obj = blocks[k]
        if kind == 'p':
            hl = _heading_level(obj)  # type: ignore[arg-type]
            if hl is not None and hl <= (lvl or 10):
                end = k - 1
                break
    return start, end, lvl

def _extract_module(blocks: List[Tuple[str, Any]], start: int, end: int) -> Optional[str]:
    """Within owner bounds, look for paragraph 'Module' followed by its value."""
    for i in range(start + 1, min(end + 1, start + 20)):  # only near the top
        kind, obj = blocks[i]
        if kind != 'p': 
            continue
        txt = (obj.text or "").strip()
        if not txt: 
            continue
        if re.match(r"^module\s*$", txt, re.I):
            # next non-empty paragraph is likely the value
            for j in range(i + 1, min(end + 1, i + 8)):
                knd2, obj2 = blocks[j]
                if knd2 != 'p': 
                    continue
                v = (obj2.text or "").strip()
                if v:
                    return v
    return None


# ------------------------ Section extraction (O/R/V only) -------------

def _collect_between(blocks, start_idx, end_idx, *, lists_only=False, include_tables=True) -> List[str]:
    """Collect non-heading lines between indices (inclusive)."""
    lines: List[str] = []
    j = start_idx
    while j <= end_idx:
        kind, obj = blocks[j]
        if kind == 'p' and _heading_level(obj) is not None:
            j += 1
            continue
        if kind == 'p':
            t = (obj.text or "").strip()
            if t and ((not lists_only) or _is_list_item(obj)):
                lines.append(t)
        else:
            if include_tables:
                for r in obj.rows:
                    cells_text = [c.text.strip() for c in r.cells]
                    line = " | ".join([x for x in cells_text if x])
                    if line:
                        lines.append(line)
        j += 1
    return lines

def extract_owner_with_sections(
    docx_path: str,
    hit_source_id: str,
    wanted_sections: List[str] = ["Overview", "Resolution", "Verification"],
    list_only_for: Dict[str, bool] = {"Overview": False, "Resolution": True, "Verification": True},
    include_tables_as_lines: bool = True,
    max_blocks_scan: int = 800,
) -> Dict[str, Any]:
    """
    Find the owner (nearest heading above the hit), then, inside that owner,
    return ONLY the specified subsections (Overview/Resolution/Verification), in order.
    """
    doc = Document(docx_path)
    blocks: List[Tuple[str, Any]] = list(iter_block_items(doc))
    block_index = build_doc_index(doc)

    root_id = _parse_source_id_root(hit_source_id)
    if not root_id or root_id not in block_index:
        return {"owner_title": None, "owner_prefix": None, "owner_level": None, "module": None, "sections": [], "bounds": [None, None]}

    hit_block = block_index[root_id]
    owner_start, owner_end, owner_level = _find_owner_bounds(blocks, hit_block, max_scan=max_blocks_scan)
    owner_title = (blocks[owner_start][1].text or "").strip() if blocks[owner_start][0] == 'p' else None
    owner_cls = _classify_owner(owner_title or "")
    module_val = _extract_module(blocks, owner_start, owner_end)

    wanted_order = {w.lower(): i for i, w in enumerate(wanted_sections)}
    found_sections: List[Dict[str, Any]] = []

    k = owner_start + 1
    while k <= owner_end:
        kind, obj = blocks[k]
        if kind == 'p':
            lvl = _heading_level(obj)  # type: ignore[arg-type]
            if lvl is not None and lvl > (owner_level or 10):
                title = (obj.text or "").strip()
                key = title.lower()
                if key in wanted_order:
                    sub_end = owner_end
                    for z in range(k + 1, owner_end + 1):
                        kind2, obj2 = blocks[z]
                        if kind2 == 'p':
                            lvl2 = _heading_level(obj2)  # type: ignore[arg-type]
                            if lvl2 is not None and lvl2 <= lvl:
                                sub_end = z - 1
                                break
                    lines = _collect_between(
                        blocks, k + 1, sub_end,
                        lists_only=bool(list_only_for.get(title, False)),
                        include_tables=include_tables_as_lines,
                    )
                    found_sections.append({
                        "title": title,
                        "level": lvl,
                        "start_index": k + 1,
                        "end_index": sub_end,
                        "lines": lines,
                    })
                    k = sub_end
        k += 1

    found_sections.sort(key=lambda s: wanted_order.get(s["title"].lower(), 999))

    return {
        "owner_title": owner_title,
        "owner_prefix": owner_cls["prefix"],
        "owner_family": owner_cls["family"],
        "owner_level": owner_level if owner_level != 10 else None,
        "module": module_val,
        "sections": found_sections,
        "bounds": [owner_start, owner_end],
    }


# ------------------------ Hit selection w/ owner preference -----------

def _infer_preferred_families_from_query(q: str) -> List[str]:
    ql = q.lower()
    prefs: List[str] = []
    if any(w in ql for w in ["container", "cntr", "equipment"]):
        prefs.append("CNTR")
    if any(w in ql for w in ["vessel", "voyage", "etb", "eta", "baplie", "coprar"]):
        prefs.append("VSL")
    if any(w in ql for w in ["api", "webhook", "oauth", "token"]):
        prefs.append("API")
    if any(w in ql for w in ["edi", "edifact", "ansi x12", "iftsta", "coarri", "315", "301", "214"]):
        prefs.append("EDI")
    # de-dup preserve order
    seen, out = set(), []
    for p in prefs:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def pick_best_hit_with_owner(
    results: List[Tuple[str, str, float]],
    docx_path: str,
    preferred_families: Optional[List[str]] = None
) -> Optional[Tuple[str, str, float, Dict[str, Any]]]:
    """
    For each hit, compute its owner bundle; prefer owners whose 'family' is in preferred_families.
    Returns (source_id, text, score, owner_bundle)
    """
    if not results:
        return None

    doc = Document(docx_path)
    blocks: List[Tuple[str, Any]] = list(iter_block_items(doc))
    block_index = build_doc_index(doc)

    def owner_for_sid(sid: str) -> Dict[str, Any]:
        root = _parse_source_id_root(sid)
        if not root or root not in block_index:
            return {"owner_title": None, "owner_family": None, "owner_prefix": None, "start": None, "end": None, "level": None}
        hit_block = block_index[root]
        start, end, lvl = _find_owner_bounds(blocks, hit_block)
        title = (blocks[start][1].text or "").strip() if blocks[start][0] == 'p' else ""
        cls = _classify_owner(title)
        return {"owner_title": title, "owner_family": cls["family"], "owner_prefix": cls["prefix"], "start": start, "end": end, "level": lvl}

    enriched = []
    for sid, text, score in results:
        ob = owner_for_sid(sid)
        enriched.append((sid, text, score, ob))

    # Prefer preferred_families if provided
    if preferred_families:
        def pref_rank(fam: Optional[str]) -> int:
            return preferred_families.index(fam) if fam in preferred_families else 999
        enriched.sort(key=lambda r: (pref_rank(r[3]["owner_family"]), -r[2]))
    else:
        enriched.sort(key=lambda r: -r[2])

    # Return top choice
    return enriched[0]


# ------------------------ JSON assembly helpers -----------------------

def results_to_json(
    query: str,
    docx_path: str,
    results: List[Tuple[str, str, float]],
    chosen_enriched: Optional[Tuple[str, str, float, Dict[str, Any]]] = None,
    owner_bundle: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "query": query,
        "docx_path": os.path.abspath(docx_path),
        "count": len(results),
        "results": [
            {"source_id": sid, "text": text, "score": float(score)}
            for sid, text, score in results
        ],
        "chosen_match": (
            {
                "source_id": chosen_enriched[0],
                "snippet": (chosen_enriched[1] or "")[:250],
                "score": float(chosen_enriched[2]),
                "owner_detected": chosen_enriched[3],
            } if chosen_enriched else None
        ),
        "owner": owner_bundle or None
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ------------------------ Orchestration --------------------------------

def main(
    query: str,
    doc_path: str,
    preferred_owner_families: Optional[List[str]] = None
) -> str:
    """
    Search, prefer owner families (CNTR/VSL/API/EDI) inferred from query (or provided),
    extract Owner + Overview/Resolution/Verification, and return JSON.
    """
    results = semantic_docx_search(
        query=query,
        docx_path=doc_path,
        threshold=DEFAULT_THRESHOLD,
        top_k=5,
        include_tables=True,
        chunk=True,
        max_chars=500,
        return_always_topk=True,
        keyword_fallback=True
    )

    if preferred_owner_families is None:
        preferred_owner_families = _infer_preferred_families_from_query(query)

    chosen = pick_best_hit_with_owner(results, doc_path, preferred_families=preferred_owner_families)

    owner_bundle = None
    if chosen:
        sid = chosen[0]
        owner_bundle = extract_owner_with_sections(
            docx_path=doc_path,
            hit_source_id=sid,
            wanted_sections=["Overview", "Resolution", "Verification"],
            list_only_for={"Overview": False, "Resolution": True, "Verification": True},
            include_tables_as_lines=True
        )

    return results_to_json(
        query, doc_path, results,
        chosen_enriched=chosen,
        owner_bundle=owner_bundle
    )


# -------------------------- CLI --------------------------------------

if __name__ == "__main__":
    query = "Shipper/consignee role swap observed on API payload for CONTAINER_ID"
    DOCX_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "Info",
        "Knowledge Base.docx"
    )
    # You can force a family preference: e.g. ["API"] or ["CNTR", "API"]
    json_str = main(query, DOCX_PATH, preferred_owner_families=None)
    print(json_str)
