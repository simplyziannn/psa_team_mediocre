import os, json
from typing import Dict, Any, List, Optional, Tuple

MODULE_SYNONYMS = {
    "container": "Container (CNTR)", "cntr": "Container (CNTR)", "cnt": "Container (CNTR)",
    "vessel": "Vessel (VS)", "vs": "Vessel (VS)",
    "edi": "EDI/API (EA)", "edi/api": "EDI/API (EA)", "ea": "EDI/API (EA)",
    # keep 'others' as Others, sub-route chosen separately
    "others": "Others", "other": "Others"
}

ROUTE_SYNONYMS = {
    "infra": "Infra/SRE", "sre": "Infra/SRE",
    "help": "Helpdesk", "helpdesk": "Helpdesk", "desk": "Helpdesk"
}

def _contacts_path() -> str:
    """
    Look for contacts.json in (priority order):
    1) Same folder as this script: processing_folder/pdf_scan/contacts.json
    2) processing_folder/resources/contacts.json
    3) processing_folder/contacts.json
    """
    cur = os.path.dirname(os.path.abspath(__file__))            # .../processing_folder/pdf_scan
    processing_folder = os.path.dirname(cur)                    # .../processing_folder

    candidates = [
        os.path.join(cur, "contacts.json"),
        os.path.join(processing_folder, "resources", "contacts.json"),
        os.path.join(processing_folder, "contacts.json"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError("contacts.json not found. Tried:\n- " + "\n- ".join(candidates))

def _load_contacts() -> List[Dict[str, Any]]:
    path = _contacts_path()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("contacts", [])

def _norm_module(q: Optional[str]) -> Optional[str]:
    if not q: return None
    key = q.strip().lower().replace(" ", "")
    return MODULE_SYNONYMS.get(key, q)

def _norm_route(r: Optional[str]) -> Optional[str]:
    if not r: return None
    key = r.strip().lower().replace(" ", "")
    return ROUTE_SYNONYMS.get(key, r)

# ---------- Original single-row lookup (kept for compatibility) ----------
def lookup_contact_row(module_query: str, route: Optional[str] = None):
    """
    Returns:
      - For normal modules: a single dict.
      - For 'Others':
          * If route is given → only that route (Infra/SRE or Helpdesk)
          * If no route → return BOTH routes as a list of dicts.
    """
    contacts = _load_contacts()
    mod = _norm_module(module_query) or module_query
    wanted = (mod or "").lower()

    # Non-"Others" modules
    if wanted != "others":
        for row in contacts:
            m = row.get("module", "")
            if not m:
                continue
            ml = m.lower()
            if ml == wanted or wanted in ml:
                if "routes" in row:
                    sub = row["routes"].get("Helpdesk", {})
                    return {"module": "Others - Helpdesk", **sub}
                return {"module": m, **{k: row.get(k, "") for k in ("manager_name", "emails", "role", "escalation_steps")}}

    # "Others" module with sub-routes
    others = next((r for r in contacts if r.get("module") == "Others"), None)
    if others and "routes" in others:
        # Case 1: route explicitly provided
        if route:
            norm_r = _norm_route(route)
            if norm_r in ("Infra/SRE", "Helpdesk"):
                sub = others["routes"].get(norm_r, {})
                return {"module": f"Others - {norm_r}", **sub}

        # Case 2: no route provided → return both routes
        return [
            {"module": f"Others - {r}", **sub}
            for r, sub in others["routes"].items()
        ]

    # Fallback
    return {"module": "Unknown", "manager_name": "", "emails": [], "role": "", "escalation_steps": []}

# ---------- New multi-row lookup (returns both for Others) ----------
def lookup_contacts(module_query: str) -> List[Dict[str, Any]]:
    """
    Returns a LIST of rows.
      - Normal modules: [single_row]
      - 'Others': [Infra/SRE_row, Helpdesk_row]  (both routes)
    """
    contacts = _load_contacts()
    mod = _norm_module(module_query) or module_query
    wanted = (mod or "").lower()

    # Non-others → return single
    if wanted != "others":
        one = lookup_contact_row(module_query)
        return [] if one.get("module") == "Unknown" else [one]

    # Others → return both routes if available
    others = next((r for r in contacts if r.get("module") == "Others"), None)
    if others and "routes" in others:
        out: List[Dict[str, Any]] = []
        for route_name in ("Infra/SRE", "Helpdesk"):
            sub = others["routes"].get(route_name, {})
            if sub:
                out.append({"module": f"Others - {route_name}", **sub})
        return out

    # Fallback: if somehow 'Others' not present, return empty
    return []

if __name__ == "__main__":
    # Example usage:
    # 1️⃣ Use lookup_contact_row(<module>, <route>)
    #    - <module> can be: "CNTR", "Vessel", "EDI", or "Others"
    #    - <route> is only used when <module> == "Others"
    #        e.g. lookup_contact_row("Others", "Infra") or ("Others", "Helpdesk")
    #
    # 2️⃣ Use lookup_contacts(<module>) if you want "Others" to return BOTH routes together.

    #row = lookup_contact_row("CNTR", None)
    row = lookup_contact_row("Others")
    print(row)