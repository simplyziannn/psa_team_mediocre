from typing import Any, Dict, Tuple
import re, json

_CNTR_RE = re.compile(r"\b([A-Z]{4}\d{7})\b", re.I)  # e.g., CMAU00000020

def extract_text_sample(payload: Any) -> Tuple[str, str]:
    """
    Extracts the best search query (and its source) from a structured or string payload.

    Handles both dicts and JSON strings safely.
    Preference order:
      1) First container id from variables.container_ids
      2) Any container id found in evidence.text_sample
      3) evidence.text_sample (trimmed)
      4) problem_statement
      5) Fallback: stringified payload
    Returns:
        (query, source)
    """
    try:
        # --- 1️⃣ Normalize to dict ---
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                # not valid JSON → treat as plain text
                return payload.strip(), "raw_string"

        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()

        if not isinstance(payload, dict):
            return str(payload).strip(), "stringified_payload"

        # --- 2️⃣ variables.container_ids ---
        vars_ = payload.get("variables") or {}
        container_ids = vars_.get("container_ids") or []
        if isinstance(container_ids, list):
            for cid in container_ids:
                s = (cid or "").strip()
                if s:
                    return s.upper(), "variables.container_ids[0]"

        # --- 3️⃣ container id inside evidence.text_sample ---
        ev = payload.get("evidence") or {}
        ts = (ev.get("text_sample") or "").strip()
        if ts:
            m = _CNTR_RE.search(ts)
            if m:
                return m.group(1).upper(), "evidence.text_sample (container_id)"

        # --- 4️⃣ evidence.text_sample (full text) ---
        if ts:
            return ts, "evidence.text_sample"

        # --- 5️⃣ problem_statement ---
        ps = (payload.get("problem_statement") or "").strip()
        if ps:
            return ps, "problem_statement"

        # --- 6️⃣ nothing ---
        return "", "none"

    except Exception as e:
        print(f"️ extract_text_sample error: {e}")
        return "", "error"
