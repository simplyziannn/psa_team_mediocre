import sys
from pprint import pprint
from excel_processing import process_alert_json, DEFAULT_EXCEL_PATH
import excel_processing as _proc

SAMPLE_ALERT = {
    "problem_statement": "EDI REF-COP-0001 stuck without ACK. Vessel: Mv Lion City 01 Api Reported Discharge With Correlation Corr- Terminal: Pasir Panjang Terminal 4 Correlation: CORR-0001",
    "incident_type": "EDI",
    "variables": {
        "cntr_no": ["MSKU0000001"],
        "message_ref": ["REF-COP-0001"],
        "error_codes": [],
        "vessel_name": [
            "MV LION CITY 01 API REPORTED DISCHARGE WITH CORRELATION CORR-"
        ],
        "voyages": [],
        "terminals": ["PASIR PANJANG TERMINAL 4"],
        "correlation_id": ["CORR-0001"],
        "is_duplicate_hint": False,
        "is_ack_missing_hint": True,
        "edi_types": ["COARRI", "COPARN"],
    },
    "evidence": {
        "flags": {
            "duplicate": False,
            "ack_missing": True,
            "baplie": False,
            "coarri": True,
            "error": False,
            "stuck": False
        },
        "text_sample": "EMAIL ALR-861770 | CORR-0001 — DISCHARGE NOT REFLECTED FOR MSKU0000001  HI OPS  FOR CONTAINER MSKU0000001 ON MV LION CITY 01  API REPORTED DISCHARGE WITH CORRELATION CORR-0001 AT 2025-10-03 17 20  BUT NO COARRI WAS GENERATED FOR THIS UNIT  THE LAST EDI ON RECORD IS COPARN REF-COP-0001  08 01   AND T"
    },
    "confidence": 0.98
}


def get_any(d, keys, default=""):
    for k in keys:
        if k in d and d.get(k):
            return str(d.get(k))
    return default


def snippet(text: str, n: int = 120) -> str:
    t = str(text or "")
    return t if len(t) <= n else t[: n - 1] + "…"


def print_matches(matches, limit=3):
    for r in matches[:limit]:
        score = r.get("_match_score")
        subj = get_any(r, ["subject", "Subject", "title", "summary"]) or "(no subject)"
        cid = get_any(r, ["case_id", "Case ID", "ticket", "Ticket ID"]) or "-"
        prod = get_any(r, ["product", "Product"]) or "-"
        date = get_any(r, ["date", "created", "opened", "Created At"]) or "-"
        desc = get_any(r, ["description", "details", "notes", "body"]) or ""
        sheet = r.get("_sheet") or "?"
        print(f"  {score} | case_id={cid} | product={prod} | date={date} | sheet={sheet}")
        if subj:
            print(f"    subject: {snippet(subj)}")
        if desc:
            print(f"    desc:    {snippet(desc)}")


def run_sample(alert=SAMPLE_ALERT):
    print("\n== Sample - EDI ACK Missing ==")
    res = process_alert_json(alert, excel_path=DEFAULT_EXCEL_PATH, max_results=10)
    la = res.get("llm_analysis", {})
    src = la.get("source")
    print(f"Problem Statement (source={src}):")
    pprint(la.get("problem_statement"))
    print("Likely cause:", la.get("likely_cause"))
    print("Priority:", la.get("priority"))
    print("Suggested escalation:", la.get("suggested_escalation"))
    print("Related cases:", la.get("related_case_ids"))
    print("Assumptions:", la.get("assumptions"))
    print("SOP (first 5 steps):")
    for step in (la.get("sop", [])[:5]):
        print(" -", step)
    if res.get("kb_fallback_used"):
        print("Note: No historical case log found. Used Knowledge Base fallback.")
    if la.get("source") == "escalation_contacts":
        esc = la.get("escalation", {})
        print("Escalation Contacts:")
        for c in esc.get("contacts", []):
            print(" -", c)
        print("Escalation Procedure:")
        for step in esc.get("procedure", []):
            print(" -", step)
    print("Top matches:")
    print_matches(res.get("matches", []), limit=3)


def run_failure_demo():
    print("\n== Failure Demo - Expecting at least 1 historical match (will fail) ==")
    nonsense_alert = {
        "problem_statement": "XQZ-DOES-NOT-EXIST random incident that should not match anything",
        "incident_type": "Unknown-Service",
        "variables": {
            "cntr_no": ["ZZZ9999999"],
            "message_ref": ["REF-NOPE-0000"],
            "correlation_id": ["CORR-NOPE-0000"],
            "edi_types": ["NOTREAL"],
        },
        "evidence": {"text_sample": "Totally unrelated gibberish code ABC-XYZ-0000"},
    }
    res = process_alert_json(nonsense_alert, excel_path=DEFAULT_EXCEL_PATH, max_results=10)
    matches = res.get("matches", [])
    if len(matches) < 1:
        print("FAILED: Expected at least 1 historical match, got 0.")
        sys.exit(1)
    else:
        print("UNEXPECTED PASS: Found matches when none expected.")
        print_matches(matches, limit=3)


def run_escalation_demo():
    print("\n== Escalation Demo - Product Team Escalation Contacts ==")
    entities = {
        "product": "EDI",
        "services": ["COARRI"],
        "keywords": ["ACK missing", "COARRI"],
        "error_codes": [],
        "case_ids": [],
        "subject": "Demo: escalate to product team",
        "summary": "Demo escalation when no history/KB",
    }
    esc = _proc._escalation_from_contacts(entities, _proc.DEFAULT_ESCALATION_PDF)
    print("Module:", esc.get("module"))
    print("Contacts:")
    for c in esc.get("contacts", []):
        print(" -", c)
    print("Procedure:")
    for step in esc.get("procedure", []):
        print(" -", step)


if __name__ == "__main__":
    try:
        if "--fail" in sys.argv or "--demo-fail" in sys.argv:
            run_failure_demo()
        elif "--escalate" in sys.argv:
            run_escalation_demo()
        else:
            print("== Running JSON alert sample ==")
            run_sample()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Test failed: {e}")
