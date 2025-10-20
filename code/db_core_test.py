from pathlib import Path
import importlib.util
import json

def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_connector(raw_email: str) -> dict:
    """
    Process a raw email text string:
    1. Extract problem info using email_processor
    2. Query seed JSONs for matching data
    Returns a dict: {
        "problem_draft": {...},
        "matches": [...],
        "error": "..." (if any)
    }
    """
    base = Path(__file__).resolve().parent  # code/
    email_path = base / 'email_processor' / 'email_processor.py'
    query_path = base / 'db_folder' / 'query_seed.py'

    # Dynamically import modules
    email_mod = load_module_from_path('email_processor', email_path)
    query_mod = load_module_from_path('query_seed', query_path)

    if not raw_email or not raw_email.strip():
        return {"error": "No input provided"}

    # Process email text into structured ProblemDraft
    try:
        pd = email_mod.process_any(raw_email.strip())
        pd_dict = pd.to_dict() if hasattr(pd, "to_dict") else (
            pd if isinstance(pd, dict) else {}
        )
    except Exception as e:
        return {"error": f"Email processing failed: {e}"}

    # Load JSON seed data
    vessels = query_mod.load_json(query_mod.FILES["vessels"])
    containers = query_mod.load_json(query_mod.FILES["containers"])
    edi = query_mod.load_json(query_mod.FILES["edi_messages"])
    api = query_mod.load_json(query_mod.FILES["api_events"])

    results = []
    vars = pd_dict.get("variables", {})

    # Extract identifiers
    cntrs = vars.get("cntr_no") or []
    msgs = vars.get("message_ref") or []
    vnames = vars.get("vessel_name") or []
    corrs = vars.get("correlation_id") or []

    # Search
    for c in cntrs:
        results += query_mod.find_by_key(containers, "cntr_no", c)
    for m in msgs:
        results += query_mod.find_by_key(edi, "message_ref", m)
    for v in vnames:
        results += query_mod.find_by_key(vessels, "vessel_name", v)
    for r in corrs:
        results += query_mod.find_by_key(api, "correlation_id", r)

    # Deduplicate
    def _id_key(obj: dict) -> str:
        for key in ["cntr_no", "message_ref", "correlation_id", "imo_no", "vessel_name"]:
            if obj.get(key):
                return f"{key}:{obj[key]}"
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    seen, unique = set(), []
    for r in results:
        k = _id_key(r)
        if k not in seen:
            seen.add(k)
            unique.append(r)

    if not unique:
        return {
            "problem_draft": pd_dict,
            "matches": [],
            "error": "No matches found for extracted variables"
        }

    return {"problem_draft": pd_dict, "matches": unique}

def main(): 
    sample = "EDI REF-COP-0001 stuck without ACK for MV Lion City 01"
    output = run_connector(sample)
    print(json.dumps(output, indent=2, ensure_ascii=False))
# Example usage (for testing):
if __name__ == "__main__":

    main()