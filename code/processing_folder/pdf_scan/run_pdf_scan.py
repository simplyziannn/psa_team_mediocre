# one_call_contacts.py
import json
from typing import List, Dict, Any, Optional, Union

#from json_extract import extract_module_from_json


if __name__ != "__main__":
    from .json_extract import extract_module_from_json
    from .contacts_lookup import lookup_contact_row, lookup_contacts
else:
    from json_extract import extract_module_from_json
    from contacts_lookup import lookup_contact_row, lookup_contacts

def resolve_alert_contacts(as_json: bool = False, route: Optional[str] = None) -> Union[List[Dict[str, Any]], str]:
    """
    One-call function:
      - Reads module from JSON OUTPUT/alert_result.json
      - Normal modules → returns [single row]
      - 'Others' (no route) → returns BOTH routes as a list
      - 'Others' + route ("Infra"/"Helpdesk") → returns [that route only]
    Set as_json=True to get a pretty JSON string.
    """
    module = extract_module_from_json()
    if not module:
        rows: List[Dict[str, Any]] = []
        return json.dumps(rows, indent=2) if as_json else rows

    mod_l = module.strip().lower()

    # Others → both routes (unless a specific route is requested)
    if mod_l == "others" or mod_l.startswith("others"):
        if route:
            one = lookup_contact_row("Others", route)
            rows = one if isinstance(one, list) else [one]
        else:
            rows = lookup_contacts("Others")
    else:
        one = lookup_contact_row(module, None)
        rows = one if isinstance(one, list) else [one]

    return json.dumps(rows, indent=2) if as_json else rows

def main():
    test=resolve_alert_contacts()
    print(test)
if __name__ == "__main__":
    main()