import sys, os, json
from datetime import datetime
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


def run_sample(alert=SAMPLE_ALERT, out_path: str | None = None):
    res = process_alert_json(alert, excel_path=DEFAULT_EXCEL_PATH, max_results=10)
    # JSON payload to save
    payload = {
        "input_alert": alert,
        "excel_path": DEFAULT_EXCEL_PATH,
        "result": res,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    # Determine output file
    if not out_path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(os.path.dirname(__file__), f"result_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Saved JSON to:", out_path)
    # Also print JSON to stdout for visibility
    print(text)


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
        # Optional: --out <file>
        out_file = None
        if "--out" in sys.argv:
            try:
                out_file = sys.argv[sys.argv.index("--out") + 1]
            except Exception:
                out_file = None
        if "--fail" in sys.argv or "--demo-fail" in sys.argv:
            run_failure_demo()
        elif "--escalate" in sys.argv:
            run_escalation_demo()
        else:
            run_sample(out_path=out_file)
    except SystemExit:
        raise
    except Exception as e:
        print(f"Test failed: {e}")

