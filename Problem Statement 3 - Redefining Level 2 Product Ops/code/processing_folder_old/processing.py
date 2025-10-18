#!/usr/bin/env python3
"""
Processing pipeline:
1) Read a JSON alert.
2) Look for similar cases in Case Log (Excel) -> use LLM to synthesize Problem, Solutions, SOP.
3) If Solutions and SOP missing -> look up Base Knowledge DOCX (Overview sections) -> LLM.
4) If still missing -> escalate via Product Team Escalation Contacts (PDF) -> LLM extracts contacts & procedure.
5) Print JSON to stdout AND save a timestamped JSON file next to this script.

Only external dependency in your repo: LLM_folder.call_openai_basic.ask_gpt5
"""

import os, re, sys, json
from pathlib import Path
from datetime import datetime,timezone,timedelta
from typing import Any, Dict, List, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----- Your LLM hook (keep as-is) -----
from LLM_folder.call_openai_basic import ask_gpt5

SGT = timezone(timedelta(hours=8))
# ----- Defaults (you can override with CLI flags) -----
REPO_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parts) >= 3 else Path.cwd()
DEFAULT_CASE_LOG = REPO_ROOT / "Info" / "Case Log.xlsx"
DEFAULT_KB_DOCX  = REPO_ROOT / "Info" / "Knowledge Base.docx"
DEFAULT_ESC_PDF  = REPO_ROOT / "Info" / "Product Team Escalation Contacts.pdf"

# =======================
# Utility / Safe Parsers
# =======================

def _to_json(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text or "")
        if m:
            try:
                obj = json.loads(m.group(0))
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def _toks(s: str) -> List[str]:
    s = _norm(s)
    s = re.sub(r"[^a-z0-9_\- ]+", " ", s)
    return [t for t in s.split() if t]
_STOPWORDS = {
    "a","an","and","or","the","of","to","for","on","at","in","is","was","were","be","with","by",
    "this","that","these","those","it","as","from","into","about","over","under","up","down",
    "your","our","their","you","we","they","i","he","she","them","us","me"
}

def _toks_domain(s: str):
    """Like _toks, but drops stopwords and very short tokens."""
    toks = _toks(s)
    return [t for t in toks if len(t) >= 3 and t not in _STOPWORDS]
# ==========================
# Excel Case Log -> matches
# ==========================

def _load_excel_first_sheet(path: Path):
    import pandas as pd
    if not path.exists():
        return None, None
    xls = pd.ExcelFile(str(path))
    sheet = xls.sheet_names[0]
    df = xls.parse(sheet)
    return df, sheet

def _canon_cols(cols: List[str]) -> Dict[str, str]:
    m = {}
    alt = {
        "case_id": {"case id","caseid","id","ticket","ticket id"},
        "subject": {"subject","title","summary"},
        "description": {"description","details","notes","body"},
        "product": {"product","service","module","product area","incident type","category"},
        "date": {"date","created","opened","created at"},
    }
    def simp(c): return re.sub(r"\s+"," ",c.strip().lower().replace("_"," "))
    for c in cols:
        sc = simp(c)
        hit=False
        for k, alts in alt.items():
            if sc in alts or sc==k:
                m[k]=c; hit=True; break
        if not hit: m.setdefault(sc, c)
    return m

def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _extract_codes(text: str) -> List[str]:
    if not text: return []
    pat = r"\b(?:ALR|INC|TCK|REF|CORR|ERR|ORA|SQL|HTTP|VR|VRR|COPARN|COARRI|BAPLIE)[-_]?[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b"
    return sorted(set(re.findall(pat, text, flags=re.IGNORECASE)), key=str.lower)
_CNTR_PAT = re.compile(r"\b[A-Z]{4}\d{7}\b")  # e.g., MSKU0000001

def _has_ops_signals(entities: dict, text_sample: str) -> bool:
    # signals from entities
    if entities.get("case_ids"): 
        return True
    if any(x for x in (entities.get("services") or []) if x and x.upper() in {"COARRI","COPARN","BAPLIE","CODECO","COPRAR"}):
        return True
    if any(_CNTR_PAT.findall(" ".join(entities.get("keywords") or []))):
        return True
    # signals from evidence text
    if _extract_codes(text_sample):
        return True
    if _CNTR_PAT.search(text_sample or ""):
        return True
    # subject/product keywords
    subj_toks = set(_toks_domain(entities.get("subject") or ""))
    ops_keywords = {"edi","coarri","coparn","baplie","codeco","coprar","container","incident","alert","error","ack","correlation","ref","api","webhook","status","latency","timeout","retry"}
    if subj_toks & ops_keywords:
        return True
    return False

def entities_from_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    variables = alert.get("variables", {}) or {}
    evidence  = alert.get("evidence", {}) or {}
    text_sample = (evidence.get("text_sample") or "") if isinstance(evidence, dict) else ""

    def get_list(*keys):
        out=[]
        for k in keys:
            v=variables.get(k)
            if isinstance(v,list): out.extend([str(x) for x in v if x])
            elif isinstance(v,str) and v: out.append(v)
        return out

    kw = []
    kw += get_list("container_ids","cntr_no","voyages","terminals","edi_types","correlation_id","message_ref","edi_refs","vessel_names","vessel_name")
    if variables.get("is_duplicate_hint"): kw.append("duplicate")
    if variables.get("is_ack_missing_hint"): kw += ["ack missing","missing ack","no ack"]
    kw += _extract_codes(text_sample)

    services=[]
    for s in get_list("edi_types"): services.append(s)
    if alert.get("incident_type"): services.append(str(alert.get("incident_type")))

    case_ids = []
    case_ids += get_list("message_ref","edi_refs","correlation_id")
    case_ids += _extract_codes(text_sample)

    subject = alert.get("problem_statement") or (alert.get("incident_type") or "Alert")

    return {
        "subject": subject,
        "product": alert.get("incident_type") or None,
        "error_codes": get_list("error_codes"),
        "services": services,
        "keywords": kw,
        "case_ids": case_ids,
        "summary": subject,
        "source_alert": alert,
    }

def find_matching_cases(entities: dict, excel_path: Path, max_results: int = 10) -> list:
    df, sheet = _load_excel_first_sheet(excel_path)
    if df is None:
        return []

    # Early bail: if there are no operational signals, skip matching entirely.
    evidence = entities.get("source_alert", {}).get("evidence", {}) if isinstance(entities.get("source_alert"), dict) else {}
    text_sample = evidence.get("text_sample") if isinstance(evidence, dict) else ""
    if not _has_ops_signals(entities, text_sample or ""):
        return []

    mp = _canon_cols(list(df.columns))
    subj = entities.get("subject") or ""
    prod = entities.get("product") or ""
    kws  = entities.get("keywords") or []
    ids  = set(_norm(x) for x in (entities.get("case_ids") or []))
    subj_kw_tokens = set(_toks_domain(" ".join([subj, *kws])))

    out = []
    for _, r in df.iterrows():
        d  = r.to_dict()
        rs = str(d.get(mp.get("subject", "subject"), ""))
        rd = str(d.get(mp.get("description", "description"), ""))
        rp = str(d.get(mp.get("product", "product"), ""))

        # Build row text and tokens (domain-only)
        text = f"{rs}\n{rd}"
        row_tokens = set(_toks_domain(text))

        sc = 0.0
        # Exact ID hit = very strong
        cid_col = mp.get("case_id")
        id_hit = bool(cid_col and _norm(str(d.get(cid_col, ""))) in ids)
        if id_hit:
            sc += 100.0

        # Product match: heavily down-weighted
        if prod and prod.lower() != "none":
            sc += 2.0 if _norm(prod) == _norm(rp) else 1.0 * _ratio(prod, rp)

        # Token overlap (after stopword pruning)
        token_overlap = len(subj_kw_tokens & row_tokens)
        sc += min(token_overlap, 10) * 1.0

        # Subject similarity
        subj_sim = _ratio(subj, rs)
        sc += 4.0 * subj_sim

        # EDI type overlap (domain words)
        edi_types_from_alert = {
            t.lower()
            for t in (entities.get("services") or [])
            if t and t.lower() != "none"
        }

        edi_types_from_alert |= {(_norm(prod) or "")} - {""}

        edi_overlap = len(edi_types_from_alert & row_tokens)



        # HARD GATE: require at least one strong signal
        strong_signal = (
            id_hit or
            token_overlap >= 1 or         # needs ≥2 real overlaps
            edi_overlap >= 1 or
            subj_sim >= 0.30 or
            (_norm(prod) and _norm(rp) and _norm(prod) == _norm(rp))   # NEW
        )
        if not strong_signal:
            continue

        if sc > 0.5:
            d["_match_score"] = round(sc, 3)
            d["_sheet"] = sheet
            out.append(d)

    out.sort(key=lambda x: x["_match_score"], reverse=True)
    return out[:max_results]


# =======================
# KB (DOCX) -> Overview
# =======================

def _read_kb_sections(docx_path: Path) -> List[Dict[str,str]]:
    try:
        from docx import Document
    except Exception:
        return []
    if not docx_path.exists(): return []
    try:
        doc = Document(str(docx_path))
        sections=[]
        current={"title":None,"paras":[]}
        for p in doc.paragraphs:
            t=(p.text or "").strip()
            if not t: continue
            style=getattr(p.style,"name","") or ""
            if style.lower().startswith("heading"):
                if current["title"] or current["paras"]: sections.append(current)
                current={"title":t,"paras":[]}
            else:
                current["paras"].append(t)
        if current["title"] or current["paras"]: sections.append(current)

        out=[]
        for s in sections:
            title=s.get("title") or "Untitled"
            paras=s.get("paras") or []
            overview=[]
            in_ov=False
            for t in paras:
                low=t.lower()
                if low.startswith("overview:") or low=="overview":
                    in_ov=True
                    ov=t.split(":",1)[-1].strip()
                    if ov: overview.append(ov)
                    continue
                if in_ov and (low.endswith(":") or low in {"steps","procedure","runbook","resolution"}):
                    break
                if in_ov: overview.append(t)
            if not overview and paras: overview=[paras[0]]
            out.append({"title":title,"overview":" ".join(overview).strip()})
        return out
    except Exception:
        return []

def kb_overview_corpus(kb_path: Path, terms: List[str], max_chars=12000) -> str:
    secs=_read_kb_sections(kb_path)
    terms_l=[t.lower() for t in (terms or []) if t]
    picks=[]
    for s in secs:
        blob=(s.get("title","")+"\n"+s.get("overview","")).lower()
        if not terms_l or any(t in blob for t in terms_l):
            picks.append(f"Title: {s.get('title','Untitled')}\nOverview: {s.get('overview','')}".strip())
        if sum(len(x) for x in picks)>=max_chars: break
    if not picks:
        for s in secs[:5]:
            picks.append(f"Title: {s.get('title','Untitled')}\nOverview: {s.get('overview','')}".strip())
            if sum(len(x) for x in picks)>=max_chars: break
    return "\n\n".join(picks)[:max_chars]

# ==========================
# Escalation (PDF) -> JSON
# ==========================

def _read_pdf_text(pdf_path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return ""
    if not pdf_path.exists(): return ""
    try:
        r=PdfReader(str(pdf_path)); out=[]
        for pg in r.pages:
            try: out.append(pg.extract_text() or "")
            except Exception: pass
        return "\n\n".join(out)
    except Exception:
        return ""

def escalation_from_pdf(pdf_path: Path, module_terms: List[str]) -> Dict[str, Any]:
    text=_read_pdf_text(pdf_path)
    if not text: return {}
    terms_l=[t.lower() for t in (module_terms or []) if t]
    chunks=re.split(r"\n\s*\n+", text)
    picks=[]
    for ch in chunks:
        cl=ch.lower()
        if not terms_l or any(t in cl for t in terms_l):
            picks.append(ch.strip())
        if len("\n\n".join(picks))>12000: break
    excerpt="\n\n".join(picks) if picks else text[:4000]

    system=(
        "You are an incident escalation assistant. From the given contacts/procedures text, "
        "extract a compact JSON with contacts and a numbered procedure. Reply with MINIFIED JSON only."
    )
    schema={"contacts":["string"],"procedure":["string"],"notes":"string|null"}
    user=f"EXCERPT:\n{excerpt}\n\nReturn JSON with fields: {json.dumps(schema)}"
    raw=ask_gpt5(user, system_prompt=system, max_completion_tokens=800)
    obj=_to_json(raw) or {}
    obj.setdefault("contacts",[]); obj.setdefault("procedure",[]); obj.setdefault("notes",None)
    obj["source"]="escalation_contacts"
    return obj

# ==========================
# LLM Stages
# ==========================

def llm_from_matches(entities: Dict[str,Any], matches: List[Dict[str,Any]]) -> Dict[str,Any]:
    # Compact context for the model (avoid dumping whole rows)
    def brief(m: Dict[str,Any]) -> str:
        keys = ["case_id","Case ID","ticket","Ticket ID","subject","Subject","title","summary","product","Product","date","created","Created At","description","details","notes","body"]
        parts=[]
        used=set()
        for k in keys:
            if k in m and k not in used and not str(k).startswith("_"):
                v=str(m.get(k,""))
                if v: parts.append(f"{k}: {v}"); used.add(k)
        return "; ".join(parts)[:600]

    top="\n".join([f"- score={m.get('_match_score')} | {brief(m)}" for m in matches[:5]]) or "- none -"

    system=(
        "You are a strict Level 2 Product Ops assistant. Using the alert and top past cases, "
        "produce JSON ONLY with: problem_statement, solutions, sop. Keep it short and specific."
    )
    schema={"problem_statement":"string","solutions":["string"],"sop":["string"]}
    user=(
        "ALERT_ENTITIES:\n"+json.dumps(entities, ensure_ascii=False)+"\n\n"
        "TOP_MATCHED_CASES:\n"+top+"\n\n"
        "Return JSON with fields: "+json.dumps(schema)
    )
    raw=ask_gpt5(user, system_prompt=system, max_completion_tokens=1500)
    obj=_to_json(raw) or {}
    obj.setdefault("problem_statement",""); obj.setdefault("solutions",[]); obj.setdefault("sop",[])
    obj["source"]="matches"
    return obj

def llm_from_kb(entities: Dict[str,Any], kb_corpus: str) -> Dict[str,Any]:
    system=(
        "You are a Product Ops runbook assistant. Using the Knowledge Base OVERVIEW excerpts below, "
        "derive a problem_statement, solutions, and sop relevant to the alert. JSON ONLY."
    )
    schema={"problem_statement":"string","solutions":["string"],"sop":["string"]}
    user=(
        "ALERT_ENTITIES:\n"+json.dumps(entities, ensure_ascii=False)+"\n\n"
        "KB_OVERVIEWS:\n"+kb_corpus+"\n\n"
        "Return JSON with fields: "+json.dumps(schema)
    )
    raw=ask_gpt5(user, system_prompt=system, max_completion_tokens=1500)
    obj=_to_json(raw) or {}
    obj.setdefault("problem_statement",""); obj.setdefault("solutions",[]); obj.setdefault("sop",[])
    obj["source"]="kb"
    return obj

# ==========================
# Main Orchestrator
# ==========================

def process_alert(
    alert: Dict[str,Any],
    case_log_path: Path = DEFAULT_CASE_LOG,
    kb_docx_path: Path = DEFAULT_KB_DOCX,
    esc_pdf_path: Path = DEFAULT_ESC_PDF,
    max_results: int = 10
) -> Dict[str,Any]:
    entities = entities_from_alert(alert)
    matches  = find_matching_cases(entities, Path(case_log_path), max_results=max_results)

    result = {
        "structured_alert": entities,
        "matches": matches,
        "excel_path": str(case_log_path),
        "kb_path": str(kb_docx_path),
        "escalation_pdf": str(esc_pdf_path),
        "timestamp": datetime.now(SGT).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "kb_used": False,
        "escalation_used": False,
        "llm_analysis": {}
    }

    # Stage 1: LLM from matches (if any)
    analysis = {}
    if matches:
        analysis = llm_from_matches(entities, matches)

    def _empty(ls: List[str]) -> bool:
        return not ls or all(not (s or "").strip() for s in ls)

    # If both solutions & sop empty -> try KB
    if (not analysis) or (_empty(analysis.get("solutions",[])) and _empty(analysis.get("sop",[]))):
        terms = (entities.get("keywords") or []) + (entities.get("services") or [])
        kb_corpus = kb_overview_corpus(Path(kb_docx_path), terms)
        if kb_corpus:
            kb_obj = llm_from_kb(entities, kb_corpus)
            result["kb_used"] = True
            # Prefer non-empty fields
            for k in ["problem_statement","solutions","sop"]:
                if _empty(analysis.get(k, [])) if isinstance(k, list) else not analysis.get(k):
                    analysis[k]=kb_obj.get(k, analysis.get(k))
        else:
            result["kb_used"] = False

    # If still BOTH solutions & sop empty -> escalate
    if _empty(analysis.get("solutions",[])) and _empty(analysis.get("sop",[])):
        module_terms = list({t for t in [(entities.get("product") or ""), *entities.get("services",[])] if t})
        esc = escalation_from_pdf(Path(esc_pdf_path), module_terms)
        result["escalation_used"] = True if esc else False
        analysis.setdefault("problem_statement", entities.get("subject") or "")
        # Put contacts/procedure into analysis for final JSON
        analysis["escalation"] = esc
        analysis.setdefault("solutions", [])
        analysis.setdefault("sop", esc.get("procedure", []))
        analysis.setdefault("suggested_escalation", ", ".join(esc.get("contacts", [])) or None)
    else:
        analysis.setdefault("suggested_escalation", None)

    result["llm_analysis"] = analysis
    return result

# ==========================
# CLI: read input + save
# ==========================

def _load_json_input(argv: List[str]) -> Dict[str,Any]:
    if "--input" in argv:
        p = Path(argv[argv.index("--input")+1])
        with open(p,"r",encoding="utf-8") as f: return json.load(f)
    if "--json" in argv:
        raw = argv[argv.index("--json")+1]
        return json.loads(raw)
    data=sys.stdin.read()
    if not data.strip():
        raise SystemExit("No JSON provided. Use --input <file>, --json '<...>' or pipe JSON via stdin.")
    return json.loads(data)

def _save_here(obj: Dict[str,Any], base_name="alert_result") -> Path:
    here = Path(__file__).resolve().parent
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(os.path.dirname(__file__),"JSON OUTPUT",f"{base_name}_{ts}.json")
    out = here /"JSON OUTPUT"/ f"{base_name}_{ts}.json"
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out_path

if __name__=="__main__":
    try:
        argv=sys.argv
        case_log = Path(DEFAULT_CASE_LOG)
        kb_docx  = Path(DEFAULT_KB_DOCX)
        esc_pdf  = Path(DEFAULT_ESC_PDF)
        if "--excel" in argv: case_log = Path(argv[argv.index("--excel")+1])
        if "--kb" in argv:    kb_docx  = Path(argv[argv.index("--kb")+1])
        if "--pdf" in argv:   esc_pdf  = Path(argv[argv.index("--pdf")+1])

        alert = _load_json_input(argv)
        res = process_alert(alert, case_log, kb_docx, esc_pdf, max_results=10)

        print(json.dumps(res, ensure_ascii=False, indent=2))
        saved = _save_here(res, base_name="alert_result")
        sys.stderr.write(f"\nSaved JSON: {saved}\n")
    except SystemExit:
        raise
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
