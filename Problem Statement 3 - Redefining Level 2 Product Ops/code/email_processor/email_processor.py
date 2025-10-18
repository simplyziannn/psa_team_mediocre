from __future__ import annotations
"""
Email Processor (module 1 of 3) — Robust Version
------------------------------------------------
Input:  subject:str | None, body:str
Output: ProblemDraft (problem_statement + variables for DB/Excel checks)

Design goals
- Rules-first (regex + heuristics) with DB-aware patterns
- Optional gazetteer enrichment (vessel names / terminal aliases)
- Optional LLM hook for low-confidence refinement (graceful fallback)
- Deterministic JSON schema for downstream modules

Import `process_email(subject, body)` from anywhere.

Author: Hackathon pipeline
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import re
import json

# ------------------------------------------------------------
# (Optional) Try to import your LLM wrapper. Falls back to None
# ------------------------------------------------------------
_ASK_GPT5 = None
try:
    # package-style path (when running with: python -m code.email_processor.email_processor)
    from code.LLM_folder.call_openai_basic import ask_gpt5 as _ASK_GPT5  # type: ignore
except Exception:
    try:
        # sibling package path (if code/ is the working dir)
        from LLM_folder.call_openai_basic import ask_gpt5 as _ASK_GPT5  # type: ignore
    except Exception:
        _ASK_GPT5 = None  # No LLM available; we'll skip refinement


# ------------------------------
# Public data model
# ------------------------------

@dataclass
class ProblemDraft:
    """What we pass to the compiler after email processing."""
    problem_statement: str
    incident_type: str  # 'ContainerData' | 'VesselAdvice' | 'EDI' | 'BAPLIE' | 'Other'
    variables: Dict[str, Any]  # keys to probe later (DB/Excel)
    evidence: Dict[str, Any]   # raw matches / snippets for audit
    confidence: float          # 0..1 rules coverage score

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------
# Normalization helpers
# ------------------------------

_RE_QUOTED = re.compile(r"^>.*$", re.MULTILINE)
_RE_SIG_BLOCK = re.compile(r"\n--+\n.*$", re.DOTALL)
_RE_TRAILER = re.compile(r"\n(?:Regards|Best|Thanks)[^\n]*\n.*$", re.IGNORECASE | re.DOTALL)
_RE_HEADERS = re.compile(r"^(From:|To:|Cc:|Subject:|Sent:).*$", re.MULTILINE | re.IGNORECASE)


def _normalize(text: str) -> str:
    if not text:
        return ""
    t = text
    t = _RE_HEADERS.sub("", t)
    t = _RE_QUOTED.sub("", t)
    t = _RE_SIG_BLOCK.sub("", t)
    t = _RE_TRAILER.sub("\n", t)
    t = re.sub(r"[\t\r]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _prep(text: str) -> str:
    """Uppercase + punctuation normalization for robust matching (keeps slash for MV/Voyage)."""
    t = _normalize(text)
    t = re.sub(r"[,\.;:()\[\]{}<>]", " ", t)
    return t.upper()


# ------------------------------
# Regex library (DB-aware patterns)
# ------------------------------

# Accept 4-letter owner + 6–8 digits (many real cntr nos have 7–8). Allow space in between and punctuation after.
CNTR_ID_RX = r"(?<![A-Z0-9])[A-Z]{4}\s*\d{6,8}(?=[^A-Z0-9]|$)"
# EDI refs like REF-COP-0001, REF-ARR-0009, REF-IFT-0007 (tolerate extra middle tokens)
EDI_REF_RX = r"\bREF-(?:COP|ARR|DEC|IFT)[A-Z-]*\d+\b"
# Message types in body (useful for EDI/BAPLIE inference)
EDI_TYPE_RX = r"\b(?:COARRI|COPARN|CODECO|IFTMIN|IFTMCS)\b"
# Error codes such as VESSEL_ERR_4
ERROR_CODE_RX = r"\b[A-Z]{2,}_[A-Z]{2,}_\d+\b"
# Voyage tokens: 2 digits + [E/N/W/S]
VOYAGE_RX = r"\b\d{2}[ENWS]\b"
# Vessel names: "MV <NAME>" optionally followed by "/07E"
VESSEL_RX = r"\bMV\s+([A-Z0-9][A-Z0-9\s\-]{1,60})(?:/(\d{2}[ENWS]))?\b"
# Terminals: Pasir Panjang Terminal 4, PPT4, T4
TERMINAL_RX = r"\b(?:PASIR\s+PANJANG\s+TERMINAL\s*\d|PPT\s*\d|T\d)\b"
# Correlation IDs like corr-0001 (your DB seeds); after _prep() they become CORR-0001
CORRELATION_RX = r"\bCORR-[A-Z0-9\-]{4,}\b"

PATTERNS = {
    "container_id": re.compile(CNTR_ID_RX),
    "edi_ref":      re.compile(EDI_REF_RX, re.IGNORECASE),
    "edi_type":     re.compile(EDI_TYPE_RX, re.IGNORECASE),
    "error_code":   re.compile(ERROR_CODE_RX),
    "voyage":       re.compile(VOYAGE_RX),
    "vessel_full":  re.compile(VESSEL_RX),
    "terminal":     re.compile(TERMINAL_RX),
    "correlation":  re.compile(CORRELATION_RX),  
}


KEYWORDS = {
    "duplicate":    re.compile(r"\bDUPLICATE|TWO\s+IDENTICAL|DUPLICATED\b", re.IGNORECASE),
    "ack_missing":  re.compile(r"\bNO\s+ACK|ACKNOWLEDG(E|MENT)\s+(NOT\s+)?SENT|ACK_AT\s+IS\s+NULL\b", re.IGNORECASE),
    "baplie":       re.compile(r"\bBAPLIE\b", re.IGNORECASE),
    "coarri":       re.compile(r"\bCOARRI\b", re.IGNORECASE),
    "error":        re.compile(r"\bERROR\b", re.IGNORECASE),
    "stuck":        re.compile(r"\bSTUCK|FAILED|BLOCKED\b", re.IGNORECASE),
}


# ------------------------------
# Optional gazetteer (DB-aware enrichment)
# ------------------------------

_GAZETTEER = {
    "vessels": set(),   # e.g., {"MV LION CITY 07", "MV MERLION 11"}
    "terminals": set(), # e.g., {"PASIR PANJANG TERMINAL 4", "PPT4", "T4"}
}

def load_gazetteer(vessels: Optional[List[str]] = None, terminals: Optional[List[str]] = None) -> None:
    """Call this once at app startup with names pulled from DB to boost matching."""
    if vessels:
        _GAZETTEER["vessels"].update(v.upper() for v in vessels)
    if terminals:
        _GAZETTEER["terminals"].update(t.upper() for t in terminals)


# ------------------------------
# Core extraction
# ------------------------------

@dataclass
class _Extraction:
    containers: List[str]
    edi_refs: List[str]
    edi_types: List[str]
    error_codes: List[str]
    voyages: List[str]
    vessel_names: List[str]
    terminals: List[str]
    correlation_ids: List[str]   
    flags: Dict[str, bool]


def _extract_rules(text_up: str) -> _Extraction:
    def uniq(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            x = x.strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # Containers: strip internal spaces e.g. "CMAU 00000020" -> "CMAU00000020"
    raw_cntrs = [re.sub(r"\s+", "", m) for m in PATTERNS["container_id"].findall(text_up)]

    # Vessel + voyage from "MV <NAME>/<VOY>"
    v_names, v_voys = [], []
    for m in PATTERNS["vessel_full"].finditer(text_up):
        nm = (m.group(1) or "").strip()
        if nm:
            nm = re.sub(r"\s+", " ", nm)
            v_names.append(f"MV {nm}")
        if m.group(2):
            v_voys.append(m.group(2))

    # Standalone voyages elsewhere
    raw_voys = v_voys + PATTERNS["voyage"].findall(text_up)

    # EDI message types (COARRI etc.)
    raw_edi_types = [m.group(0).upper() for m in PATTERNS["edi_type"].finditer(text_up)]

    ext = _Extraction(
        containers=uniq(raw_cntrs),
        edi_refs=uniq(PATTERNS["edi_ref"].findall(text_up)),
        edi_types=uniq(raw_edi_types),
        error_codes=uniq(PATTERNS["error_code"].findall(text_up)),
        voyages=uniq(raw_voys),
        vessel_names=uniq(v_names),
        terminals=uniq([m.group(0).strip() for m in PATTERNS["terminal"].finditer(text_up)]),
        correlation_ids=uniq(PATTERNS["correlation"].findall(text_up)),  # <-- add this
        flags={k: bool(rx.search(text_up)) for k, rx in KEYWORDS.items()},
    )


    # Gazetteer enrichment (if loaded)
    for v in _GAZETTEER["vessels"]:
        if v in text_up and v not in ext.vessel_names:
            ext.vessel_names.append(v)
    for t in _GAZETTEER["terminals"]:
        if t in text_up and t not in ext.terminals:
            ext.terminals.append(t)

    # Deduplicate again after enrichment
    ext.vessel_names = sorted(set(ext.vessel_names))
    ext.terminals = sorted(set(ext.terminals))
    return ext


# ------------------------------
# Incident typing & statement crafting (rule-based)
# ------------------------------

def _infer_incident_type(x: _Extraction) -> str:
    # Strong signals first
    if x.flags.get("baplie") or ("COARRI" in x.edi_types and not x.edi_refs and not x.error_codes):
        return "BAPLIE"
    if x.edi_refs or x.flags.get("ack_missing") or x.edi_types:
        return "EDI"
    if x.error_codes or x.vessel_names:
        return "VesselAdvice"
    if x.containers or x.flags.get("duplicate"):
        return "ContainerData"
    return "Other"


def _build_problem_statement(subj: str, body: str, x: _Extraction) -> str:
    s = _prep((subj or "") + ". " + (body or ""))
    parts: List[str] = []

    # Prefer explicit patterns
    if x.containers and x.flags.get("duplicate"):
        parts.append(f"Duplicate container information reported for {x.containers[0]}.")
    elif x.edi_refs and x.flags.get("ack_missing"):
        parts.append(f"EDI {x.edi_refs[0]} stuck without ACK.")
    elif x.error_codes:
        parts.append(f"System error {x.error_codes[0]} encountered.")
    elif x.flags.get("stuck"):
        parts.append("Process appears stuck with no acknowledgement.")

    if not parts:
        # fallback: first 140 chars of cleaned text (re-humanize a bit: capitalize)
        fallback = s[:140] + ("…" if len(s) > 140 else "")
        parts.append(fallback.capitalize())

    if x.vessel_names:
        parts.append(f"Vessel: {x.vessel_names[0].title()}")
    if x.voyages:
        parts.append(f"Voyage: {x.voyages[0]}")
    if x.terminals:
        parts.append(f"Terminal: {x.terminals[0].title()}")
    if x.correlation_ids:
        parts.append(f"Correlation: {x.correlation_ids[0]}")


    return " ".join(parts)


# ------------------------------
# Optional LLM refinement
# ------------------------------

def _llm_refine(statement: str, variables: dict) -> str:
    """If the wrapper is available, ask GPT-5-mini to polish the statement (no hallucinations)."""
    if _ASK_GPT5 is None:
        return statement  # No LLM available

    sys_prompt = (
        "You are a PSA operations assistant. "
        "Given extracted incident info, rewrite the problem statement as ONE clear sentence. "
        "Do NOT invent facts or add data not present in the payload."
    )
    user_message = json.dumps({
        "draft_statement": statement,
        "variables": variables
    }, indent=2)

    try:
        refined = _ASK_GPT5(user_message, system_prompt=sys_prompt, max_completion_tokens=120)
        refined = (refined or "").strip()
        if not refined or refined.startswith("[HTTP") or refined.startswith("[Network"):
            return statement
        # Ensure one concise sentence; truncate if the model returns more.
        refined = refined.split("\n")[0].strip()
        if not refined.endswith("."):
            refined += "."
        return refined
    except Exception:
        return statement


# ------------------------------
# Public API
# ------------------------------

def process_email(subject: Optional[str], body: str) -> ProblemDraft:
    """Parse email into ProblemDraft for downstream DB/Excel checks."""
    norm_subject = (subject or "").strip()
    norm_body = (body or "").strip()

    combined_up = _prep(f"{norm_subject}. {norm_body}")
    x = _extract_rules(combined_up)

    incident_type = _infer_incident_type(x)

    variables: Dict[str, Any] = {
        "cntr_no": x.containers,
        "message_ref": x.edi_refs,
        "error_codes": x.error_codes,
        "vessel_name": x.vessel_names,   # keep as UPPER; downstream may .upper() for joins
        "voyages": x.voyages,
        "terminals": x.terminals,
        "correlation_id": x.correlation_ids,
        # convenience booleans to drive probes
        "is_duplicate_hint": x.flags.get("duplicate", False),
        "is_ack_missing_hint": x.flags.get("ack_missing", False),
        # optional additional signal for DB/Excel rules
        "edi_types": x.edi_types,
    }

    # Coverage score: how many useful fields did we populate?
    slots = [x.containers, x.edi_refs, x.error_codes, x.vessel_names, x.voyages, x.terminals, x.edi_types]
    filled = sum(1 for s in slots if s)
    confidence = min(1.0, 0.2 + 0.14 * filled + (0.08 if any(x.flags.values()) else 0))

    statement = _build_problem_statement(norm_subject, norm_body, x)

    # LLM refinement on low confidence (if available)
    if confidence < 0.6:
        # print("⚠️  Low confidence — refining with GPT-5-mini...")  # optional log
        statement = _llm_refine(statement, variables)

    evidence = {
        "flags": x.flags,
        "text_sample": combined_up[:300],
    }

    return ProblemDraft(
        problem_statement=statement,
        incident_type=incident_type,
        variables=variables,
        evidence=evidence,
        confidence=round(confidence, 2),
    )


# ------------------------------
# Convenience wrappers
# ------------------------------

def process_any(payload: Any) -> ProblemDraft:
    """
    Accepts many input shapes and routes to process_email(subject, body).

    Supported forms:
      - str                          -> body only (subject = "")
      - (subject, body) tuple/list   -> uses both
      - dict                         -> uses common keys:
           subject: 'subject' | 'title' | 'subj'
           body:    'body' | 'text' | 'message' | 'content'
           fallback: if only one text-like field exists, it's treated as body
      - objects with attributes      -> tries .subject / .body / .text

    Returns:
      ProblemDraft
    """
    subj, body = "", ""

    # 1) String payload: treat as body-only (e.g., SMS)
    if isinstance(payload, str):
        body = payload

    # 2) Tuple/list: (subject, body) or (body,)
    elif isinstance(payload, (tuple, list)):
        if len(payload) == 0:
            pass
        elif len(payload) == 1:
            body = str(payload[0]) if payload[0] is not None else ""
        else:
            subj = "" if payload[0] is None else str(payload[0])
            body = "" if payload[1] is None else str(payload[1])

    # 3) Dict: look for common keys
    elif isinstance(payload, dict):
        # Subject candidates
        for k in ("subject", "title", "subj"):
            if k in payload and payload[k] is not None:
                subj = str(payload[k])
                break
        # Body candidates
        for k in ("body", "text", "message", "content"):
            if k in payload and payload[k] is not None:
                body = str(payload[k])
                break
        # Fallback: if no explicit body key but exactly one text-like field, use it
        if not body:
            textish_keys = [k for k, v in payload.items()
                            if isinstance(v, (str, bytes)) and k.lower() not in {"subject", "title", "subj"}]
            if len(textish_keys) == 1:
                body = str(payload[textish_keys[0]])

    # 4) Objects with attributes (e.g., email models)
    else:
        subj = _safe_attr(payload, "subject") or _safe_attr(payload, "title") or ""
        body = (_safe_attr(payload, "body")
                or _safe_attr(payload, "text")
                or _safe_attr(payload, "message")
                or _safe_attr(payload, "content")
                or "")

    return process_email(subj, body)


def _safe_attr(obj: Any, name: str) -> Optional[str]:
    try:
        val = getattr(obj, name, None)
        if val is None:
            return None
        return str(val)
    except Exception:
        return None


def process_many(items: List[Any]) -> List[Dict[str, Any]]:
    """
    Batch helper: returns a list of ProblemDraft dicts.
    Example:
        results = process_many([
            "SMS: REF-IFT-0007 stuck...",
            {"subject": "INC-1", "body": "CMAU0000012 duplicate..."},
            ("EDI alert", "REF-COP-0001 ...")
        ])
    """
    out = []
    for it in items:
        try:
            out.append(process_any(it).to_dict())
        except Exception as e:
            out.append({"error": f"failed to process item: {e}", "input_preview": str(it)[:160]})
    return out


if __name__ == "__main__":
    # ----------------------------------------------
    # How to use:
    # Call process_email(subject, body) or process_any(payload)
    # Example:
    #   result = process_any("EDI REF-IFT-0007 stuck for CMAU0000001 at PPT4")
    #   print(json.dumps(result.to_dict(), indent=2))
    # ----------------------------------------------
    sample = "COARRI REF-ARR-0013 not acked for OOLU0000013 at PPT4."
    result = process_any(sample)
    print(json.dumps(result.to_dict(), indent=2))



"""if __name__ == "__main__":
    tests: List[Tuple[str, str]] = [
        # CASE 1 — Duplicate Container
        ("Email ALR-861600 | CMAU00000020 - Duplicate Container information received",
         "Hi Jen, Please assist in checking container CMAU00000020. Customer on PORTNET is seeing 2 identical containers information."),
        # CASE 2 — VESSEL_ERR_4 / Vessel Advice
        ("Email ALR-861631 | VESSEL_ERR_4 - System Vessel Name has been used by other vessel advice",
         "Customer reported unable to create vessel advice for MV Lion City 07/07E and hit error VESSEL_ERR_4. The local vessel name had been used by other vessel advice."),
        # CASE 3 — EDI stuck / No ACK
        ("SMS INC-154599",
         "EDI message REF-IFT-0007 stuck in ERROR status (Sender: LINE-PSA, Recipient: PSA-TOS, State: No acknowledgment sent, ack_at is NULL)."),
        # CASE 4 — BAPLIE inconsistency
        ("Call TCK-742311",
         "BAPLIE inconsistency for MV PACIFIC DAWN/07E at Pasir Panjang Terminal 4: COARRI shows load completed for bay 14, but BAPLIE still lists units in those slots."),
        # CASE 5 — corr-0001 + Container + EDI + API Event (real IDs from DB)
        ("Email ALR-861770 | corr-0001 — DISCHARGE not reflected for MSKU0000001",
        "Hi Ops, for container MSKU0000001 on MV LION CITY 01, API reported DISCHARGE with correlation corr-0001 at 2025-10-03 17:20, "
        "but no COARRI was generated for this unit. The last EDI on record is COPARN REF-COP-0001 (08:01), "
        "and the box still shows as TRANSHIP at Pasir Panjang Terminal 4. "
        "Please check COARRI acknowledgment (ack_at is NULL) and EDI/API sync."),
    ]

    for i, (subj, body) in enumerate(tests, 1):
        result = process_email(subj, body)
        print(f"\n---- CASE {i} ----")
        print(json.dumps(result.to_dict(), indent=2))
"""