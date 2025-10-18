import sys
from pprint import pprint
from excel_processing import process_alert_json, DEFAULT_EXCEL_PATH

SAMPLE_ALERTS = [
    {
        "name": "ACK missing (COARRI not generated)",
        "alert": {
            "problem_statement": "EDI REF-COP-0001 stuck without ACK. Vessel: Mv Lion City 01 Api Reported Discharge With Correlation Corr- Terminal: Pasir Panjang Terminal 4 Correlation: CORR-0001",
            "incident_type": "EDI",
            "variables": {
                "cntr_no": ["MSKU0000001"],
                "message_ref": ["REF-COP-0001"],
                "error_codes": [],
                "vessel_name": ["MV LION CITY 01 API REPORTED DISCHARGE WITH CORRELATION CORR-"],
                "voyages": [],
                "terminals": ["PASIR PANJANG TERMINAL 4"],
                "correlation_id": ["CORR-0001"],
                "is_duplicate_hint": False,
                "is_ack_missing_hint": True,
                "edi_types": ["COARRI", "COPARN"],
            },
            "evidence": {
                "flags": {"duplicate": False, "ack_missing": True, "coarri": True},
                "text_sample": (
                    "ALR-861770 | CORR-0001 — DISCHARGE NOT REFLECTED FOR MSKU0000001. "
                    "COPARN REF-COP-0001 present, COARRI missing."
                ),
            },
            "confidence": 0.98,
        },
    },
    {
        "name": "Possible duplicate COARRI",
        "alert": {
            "problem_statement": "Duplicate COARRI suspected for MSKU0000002 at Pasir Panjang Terminal 3",
            "incident_type": "EDI",
            "variables": {
                "cntr_no": ["MSKU0000002"],
                "message_ref": ["REF-COA-0021"],
                "correlation_id": ["CORR-0002"],
                "terminals": ["PASIR PANJANG TERMINAL 3"],
                "edi_types": ["COARRI"],
                "is_duplicate_hint": True,
                "is_ack_missing_hint": False,
            },
            "evidence": {
                "flags": {"duplicate": True, "coarri": True},
                "text_sample": "ALR-861771 | Possible duplicate COARRI for MSKU0000002, CORR-0002",
            },
            "confidence": 0.9,
        },
    },
    {
        "name": "Non-EDI: Vessel Advice service latency",
        "alert": {
            "problem_statement": "Vessel Advice API latency spike affecting schedule lookups for MV PACIFIC DAWN",
            "incident_type": "Vessel Advice",
            "variables": {
                "vessel_name": ["MV PACIFIC DAWN"],
                "terminals": ["PASIR PANJANG TERMINAL 4"],
                "error_codes": ["HTTP 504"],
            },
            "evidence": {
                "flags": {"error": True},
                "text_sample": "ALR-861780 | 504s observed on /vessel-advice/schedule for MV PACIFIC DAWN",
            },
            "confidence": 0.7,
        },
    },
]


def print_matches(matches, limit=3):
    def get_any(d, keys, default=""):
        for k in keys:
            if k in d and d.get(k):
                return str(d.get(k))
        return default

    for r in matches[:limit]:
        score = r.get("_match_score")
        subj = get_any(r, ["subject", "Subject", "title", "summary"]) or "(no subject)"
        cid = get_any(r, ["case_id", "Case ID", "ticket", "Ticket ID"]) or "-"
        prod = get_any(r, ["product", "Product"]) or "-"
        print(f"  {score} | case_id={cid} | product={prod} | {subj}")


def run_samples():
    for i, item in enumerate(SAMPLE_ALERTS, start=1):
        print("\n== Sample", i, f"- {item['name']} ==")
        res = process_alert_json(item["alert"], excel_path=DEFAULT_EXCEL_PATH, max_results=10)
        src = res.get("llm_analysis", {}).get("source")
        print(f"Problem Statement (source={src}):")
        pprint(res.get("llm_analysis", {}).get("problem_statement"))
        print("SOP (first 5 steps):")
        for step in (res.get("llm_analysis", {}).get("sop", [])[:5]):
            print(" -", step)
        if res.get("kb_fallback_used"):
            print("Note: No historical case log found. Used Knowledge Base fallback.")
        print("Top matches:")
        print_matches(res.get("matches", []), limit=3)


def run_failure_demo():
    """Intentionally fail: require at least one historical match for a nonsense alert."""
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


if __name__ == "__main__":
    try:
        if "--fail" in sys.argv or "--demo-fail" in sys.argv:
            run_failure_demo()
        else:
            print("== Running JSON alert sample suite ==")
            run_samples()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Test failed: {e}")
