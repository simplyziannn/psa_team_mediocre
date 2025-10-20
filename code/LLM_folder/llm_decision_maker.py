# code/LLM_folder/llm_decision_maker.py
from __future__ import annotations
import json, pickle, os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from LLM_folder.embedding_helper import get_embedding, get_cosine_similarity
from LLM_folder.call_openai_basic import ask_gpt5

# --------------------------------------------------------------------------------------
# PATHS (no hardcoding) — relative to this file, with ENV overrides
# --------------------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
CODE_DIR = THIS_FILE.parents[1]               # .../code
PROC_DIR = CODE_DIR / "processing_folder"     # .../code/processing_folder

PROMPT_PATH = Path(os.getenv("PROMPT_CONFIG_PATH", THIS_FILE.with_name("prompt_config.json")))
CONTACTS_JSON_PATH = Path(os.getenv("CONTACTS_JSON_PATH", PROC_DIR / "pdf_scan" / "contacts.json"))
DOCX_EMB_PATH = Path(os.getenv("DOCX_EMB_PATH", PROC_DIR / "doc_scan" / ".emb_cache" / "embeddings.pkl"))
EXCEL_EMB_PATH = Path(os.getenv("EXCEL_EMB_PATH", PROC_DIR / "excel_scan" / "case_log.embcache.pkl"))

# --------------------------------------------------------------------------------------
# Helpers: map/loader for escalation catalogs
# --------------------------------------------------------------------------------------
def _map_module_to_target(module: str) -> str:
    """Map arbitrary module names from contacts.json/DB to allowed target labels."""
    if not module:
        return "Others"
    m = module.strip().upper()
    if "EDI" in m or "API" in m:
        return "EDI/API"
    if "CONTAINER" in m or "CNTR" in m:
        return "Container (CNTR)"
    if "VESSEL" in m or m == "VS":
        return "Vessel (VS)"
    return "Others"

def _load_contacts_catalog(path: Path) -> List[Dict[str, Any]]:
    """
    Load /pdf_scan/contacts.json and normalize to:
      [{ "target": <allowed>, "contacts": [{"name","email"}], "steps": [str, ...] }, ...]
    Supports nested 'Others.routes' blocks if present.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    contacts = raw.get("contacts") if isinstance(raw, dict) else []
    if not isinstance(contacts, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in contacts:
        if not isinstance(item, dict):
            continue

        # Nested routes under "Others"
        if item.get("module", "").strip().lower() == "others" and isinstance(item.get("routes"), dict):
            for _, route in item["routes"].items():
                if not isinstance(route, dict):
                    continue
                tgt = "Others"
                c_list = []
                name = route.get("manager_name") or route.get("name")
                emails = route.get("emails") if isinstance(route.get("emails"), list) else []
                if name and emails:
                    c_list.append({"name": str(name), "email": str(emails[0])})
                steps = route.get("escalation_steps") if isinstance(route.get("escalation_steps"), list) else []
                out.append({"target": tgt, "contacts": c_list, "steps": [str(s) for s in steps]})
            continue

        # Flat record
        tgt = _map_module_to_target(item.get("module", ""))
        c_list = []
        name = item.get("manager_name") or item.get("name")
        emails = item.get("emails") if isinstance(item.get("emails"), list) else []
        if name and emails:
            c_list.append({"name": str(name), "email": str(emails[0])})
        steps = item.get("escalation_steps") if isinstance(item.get("escalation_steps"), list) else []
        out.append({"target": tgt, "contacts": c_list, "steps": [str(s) for s in steps]})

    return out

# --------------------------------------------------------------------------------------
# CACHE LOADERS (normalized to [(label, embedding)])
# --------------------------------------------------------------------------------------
def load_docx_embedding_cache(path: Path) -> List[Tuple[str, List[float]]]:
    """DOCX cache: dict { <hash_id:str> : <embedding:list[float]> } -> [(id, emb)]."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected DOCX cache type: {type(data)}")

    normalized: List[Tuple[str, List[float]]] = []
    for hash_id, emb in data.items():
        if isinstance(emb, (list, tuple)):
            normalized.append((str(hash_id), list(emb)))
    if not normalized:
        raise ValueError("DOCX cache parsed but contained no (id, embedding) pairs.")
    return normalized

def load_excel_embedding_cache(path: Path) -> List[Tuple[str, List[float]]]:
    """Excel cache: dict with 'items' -> each has 'embedding' and some text -> [(label, emb)]."""
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "items" not in data:
        raise ValueError(f"Unexpected Excel cache structure: {type(data)}, keys {list(data)[:10]}")

    items = data.get("items") or []
    normalized: List[Tuple[str, List[float]]] = []

    for it in items:
        if not isinstance(it, dict):
            continue
        emb = it.get("embedding") or it.get("vector") or it.get("emb") or it.get("vec")
        if not isinstance(emb, (list, tuple)):
            continue
        text = (
            it.get("text") or it.get("value") or it.get("cell_text")
            or it.get("content") or it.get("preview") or it.get("snippet")
        )
        if not text:
            sheet = it.get("sheet") or it.get("sheet_name")
            r = it.get("row") or it.get("r")
            c = it.get("col") or it.get("c") or it.get("column")
            bits = []
            if sheet: bits.append(str(sheet))
            if r is not None: bits.append(f"r{r}")
            if c is not None: bits.append(f"c{c}")
            text = "[" + " ".join(bits) + "]" if bits else str(it)
        normalized.append((str(text), list(emb)))

    if not normalized:
        raise ValueError("Excel cache parsed but found no items with embeddings.")
    return normalized

# --------------------------------------------------------------------------------------
# SEMANTIC SEARCH + SUMMARIZER
# --------------------------------------------------------------------------------------
def semantic_search(query: str, cache: List[Tuple[str, List[float]]], top_k: int = 3) -> List[Tuple[str, float]]:
    """Similarity search against a cache of [(label, embedding)]."""
    query_emb = get_embedding(query)
    results: List[Tuple[str, float]] = []
    for label, emb in cache:
        try:
            score = get_cosine_similarity(query_emb, emb)
            results.append((label, score))
        except Exception as e:
            print(f"️ Skipped one cache entry: {e}")
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def summarize_references(name: str, matches: List[Tuple[str, float]]) -> str:
    if not matches:
        return f"[{name}] No relevant entries found."
    lines = [f"[{name}] Top {len(matches)}:"]
    for label, score in matches:
        short = (label or "").replace("\n", " ")[:300]
        lines.append(f"- ({score:.3f}) {short}")
    return "\n".join(lines)

# --------------------------------------------------------------------------------------
# POST-PARSE GUARDRAILS
# --------------------------------------------------------------------------------------
def _ensure_schema(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing keys with safe defaults and coerce types."""
    if not isinstance(decision, dict):
        return {"raw_response": str(decision)}

    decision.setdefault("module", "unknown")
    decision.setdefault("summary", "unknown")
    decision.setdefault("root_cause", "unknown")

    # resolution_steps -> list[str]
    steps = decision.get("resolution_steps", [])
    if isinstance(steps, str):
        steps = [steps]
    if not isinstance(steps, list):
        steps = []
    decision["resolution_steps"] = [str(s) for s in steps]

    # escalation -> object with target/contacts/steps
    esc = decision.get("escalation")
    if not isinstance(esc, dict):
        esc = {}
    esc.setdefault("target", decision.get("escalation_target", "unknown"))
    contacts = esc.get("contacts", [])
    if not isinstance(contacts, list):
        contacts = []
    esc["contacts"] = [{"name": str(c.get("name","unknown")), "email": str(c.get("email","unknown"))}
                       for c in contacts if isinstance(c, dict)]
    esc_steps = esc.get("steps", [])
    if isinstance(esc_steps, str):
        esc_steps = [esc_steps]
    if not isinstance(esc_steps, list):
        esc_steps = []
    esc["steps"] = [str(s) for s in esc_steps]
    decision["escalation"] = esc

    # evidence_used -> list[str]
    ev = decision.get("evidence_used", [])
    if isinstance(ev, str):
        ev = [ev]
    if not isinstance(ev, list):
        ev = []
    decision["evidence_used"] = [str(x) for x in ev]

    # confidence -> float
    try:
        decision["confidence"] = float(decision.get("confidence", 0.0))
    except Exception:
        decision["confidence"] = 0.0

    # cleanup legacy
    decision.pop("escalation_target", None)
    return decision

def _enforce_escalation(decision: Dict[str, Any],
                        allowed: List[str],
                        catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Force escalation.target ∈ allowed and copy contacts/steps verbatim from catalog/DB."""
    if not isinstance(decision, dict):
        return decision

    esc = decision.get("escalation") or {}
    target = (esc.get("target") or "unknown").strip()

    # If target missing/invalid, infer by module; else default to 'Others'
    if target not in allowed:
        mod = (decision.get("module") or "").upper()
        if "EDI" in mod:
            target = "EDI/API"
        elif "CONTAINER" in mod or "CNTR" in mod:
            target = "Container (CNTR)"
        elif "VESSEL" in mod or "VS" in mod:
            target = "Vessel (VS)"
        else:
            target = "Others"

    # Pull canonical contacts/steps from catalog
    canon = next((c for c in catalog if isinstance(c, dict) and c.get("target") == target), None)
    if canon:
        contacts = canon.get("contacts", [])
        steps = canon.get("steps", [])
    else:
        contacts, steps = [], []

    decision["escalation"] = {
        "target": target,
        "contacts": contacts if isinstance(contacts, list) else [],
        "steps": [str(s) for s in steps] if isinstance(steps, list) else []
    }
    return decision

# --------------------------------------------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------------------------------------------
def decide_solution(
    compiled_json_path: str | Path,
    model: str = "gpt-5-mini",
    raw_email: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reads compiled_output.json, runs semantic lookups on DOCX & Excel caches,
    and asks the LLM (via Azure APIM helper ask_gpt5) to synthesize a final decision.
    """
    compiled_path = Path(compiled_json_path)
    data = json.loads(compiled_path.read_text(encoding="utf-8"))

    # 1) Problem text (query for semantic search)
    query = (
        data.get("alert_result", {}).get("query")
        or str(data.get("db_result", {}))
        or "Unknown problem context"
    )

    # 2) Load caches
    print(" Loading embedding caches ...")
    doc_cache = load_docx_embedding_cache(DOCX_EMB_PATH)
    excel_cache = load_excel_embedding_cache(EXCEL_EMB_PATH)

    # 3) Prompt config + similarity threshold + allowed escalations
    prompt_cfg = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    sim_th = float(prompt_cfg.get("similarity_threshold", 0.70))
    allowed_escalations = prompt_cfg.get("allowed_escalations", [])

    # 3a) Semantic search (filter by threshold)
    raw_doc = semantic_search(query, doc_cache, top_k=6)
    raw_xls = semantic_search(query, excel_cache, top_k=6)
    top_doc = [(lbl, sc) for (lbl, sc) in raw_doc if sc >= sim_th][:3]
    top_excel = [(lbl, sc) for (lbl, sc) in raw_xls if sc >= sim_th][:3]

    # 3b) Build escalation catalog from three sources (contacts.json -> DB -> prompt config)
    pdf_catalog = _load_contacts_catalog(CONTACTS_JSON_PATH)  # strongest source
    db_catalog_raw = []
    db_res = data.get("db_result", {}) or {}
    for k in ("escalations", "escalation_catalog", "contacts", "owners", "owner_contacts"):
        v = db_res.get(k)
        if isinstance(v, list):
            db_catalog_raw = v
            break

    # Normalize DB records -> {target, contacts, steps}
    db_catalog: List[Dict[str, Any]] = []
    for item in db_catalog_raw:
        if not isinstance(item, dict):
            continue
        tgt = _map_module_to_target(item.get("target") or item.get("module") or "Others")
        c_list = []
        name = item.get("manager_name") or item.get("name")
        emails = item.get("emails") if isinstance(item.get("emails"), list) else []
        if name and emails:
            c_list.append({"name": str(name), "email": str(emails[0])})
        steps = item.get("escalation_steps") if isinstance(item.get("escalation_steps"), list) else []
        db_catalog.append({"target": tgt, "contacts": c_list, "steps": [str(s) for s in steps]})

    # Normalize CFG records -> {target, contacts, steps}
    cfg_catalog_raw = prompt_cfg.get("escalation_catalog", [])
    cfg_catalog: List[Dict[str, Any]] = []
    for c in cfg_catalog_raw or []:
        if isinstance(c, dict) and "target" in c:
            cfg_catalog.append({
                "target": _map_module_to_target(c.get("target")),
                "contacts": c.get("contacts", []),
                "steps": [str(s) for s in (c.get("steps") or [])]
            })

    # Merge with precedence: PDF > DB > CFG
    catalog_by_target: Dict[str, Dict[str, Any]] = {}
    for src in (cfg_catalog, db_catalog, pdf_catalog):
        for c in src:
            t = c.get("target")
            if not t:
                continue
            catalog_by_target[t] = c  # later sources overwrite earlier ones

    # Ensure entries exist for all allowed targets
    for tgt in allowed_escalations:
        if tgt not in catalog_by_target:
            catalog_by_target[tgt] = {"target": tgt, "contacts": [], "steps": []}

    escalation_catalog = list(catalog_by_target.values())

    # 4) Build prompt from config
    task_lines = prompt_cfg.get("task_prompt", [])
    task_text = "\n".join(task_lines)

    # Include original raw email at the top (PRIMARY ground truth in the prompt)
    orig_block = f"\n=== ORIGINAL CONTENT ===\n{raw_email}\n" if raw_email else ""

    context = f"""
{orig_block}
=== ALERT SUMMARY ===
{json.dumps(data.get("alert_result", {}), indent=2)}

=== DB RESULT ===
{json.dumps(data.get("db_result", {}), indent=2)}

{summarize_references("DOCX", top_doc)}

{summarize_references("EXCEL", top_excel)}

### ALLOWED ESCALATIONS
{json.dumps(allowed_escalations, indent=2)}

### ESCALATION CATALOG (copy CONTACTS and STEPS verbatim from the chosen target)
{json.dumps(escalation_catalog, indent=2)}

### TASK
{task_text}
""".strip()

    # 5) Ask LLM via APIM helper
    system_prompt = prompt_cfg.get("system_prompt", "You are a PSA L2 resolution assistant. Reply ONLY with JSON.")
    text_out = ask_gpt5(
        user_message=context,
        system_prompt=system_prompt,
        max_completion_tokens=4000
    )

    # 6) Parse JSON or fallback, then enforce schema & escalation
    try:
        decision = json.loads(text_out)
    except Exception:
        decision = {"raw_response": text_out}

    # Traceability: keep the original content that grounded this decision
    if isinstance(decision, dict) and raw_email:
        decision.setdefault("_trace", {})["original_content"] = raw_email

    # Guardrails: ensure schema + enforce escalation
    decision = _ensure_schema(decision)
    decision = _enforce_escalation(decision, allowed_escalations, escalation_catalog)

    # 7) Save JSON
    out_path = compiled_path.with_name("final_solution.json")
    out_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f" Final structured solution saved to: {out_path}")

    # 8) Create human-readable summary (uses enforced escalation block)
    human_lines: List[str] = []
    human_lines.append("INCIDENT SUMMARY")
    human_lines.append("=" * 80)
    human_lines.append(f"Module: {decision.get('module', 'N/A')}")
    human_lines.append(f"\nSummary:\n{decision.get('summary', 'N/A')}\n")
    human_lines.append(f"Root Cause:\n{decision.get('root_cause', 'N/A')}\n")

    steps = decision.get("resolution_steps") or []
    if isinstance(steps, list) and steps:
        human_lines.append("Resolution Steps:")
        for s in steps:
            human_lines.append(f" - {str(s).strip()}")
    elif isinstance(steps, str):
        human_lines.append(f"Resolution Steps:\n{steps}")

    esc = decision.get("escalation", {}) or {}
    human_lines.append(f"\nEscalation Target: {esc.get('target', 'N/A')}")
    contacts = esc.get("contacts") or []
    if contacts:
        human_lines.append("Contacts:")
        for c in contacts:
            human_lines.append(f" - {c.get('name','unknown')} <{c.get('email','unknown')}>")
    esc_steps = esc.get("steps") or []
    if esc_steps:
        human_lines.append("Escalation Steps:")
        for s in esc_steps:
            human_lines.append(f" - {str(s).strip()}")

    human_lines.append("=" * 80)
    human_readable = "\n".join(human_lines)

    # Save text version
    human_path = compiled_path.with_name("final_solution_human.txt")
    human_path.write_text(human_readable, encoding="utf-8")
    print(f" Human-readable summary saved to: {human_path}")

    # 9) Return both
    return {"json": decision, "text": human_readable}
