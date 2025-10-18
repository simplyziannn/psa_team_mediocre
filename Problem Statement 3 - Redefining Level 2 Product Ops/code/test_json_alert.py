import json
from pprint import pprint

from excel_processing import process_alert_json, DEFAULT_EXCEL_PATH


SAMPLE = {
    "problem_statement": "EDI REF-COP-0001 stuck without ACK. Vessel: Mv Lion City 01 Api Reported Discharge With Correlation Corr- Terminal: Pasir Panjang Terminal 4",
    "incident_type": "EDI",
    "variables": {
        "container_ids": ["MSKU0000001"],
        "edi_refs": ["REF-COP-0001"],
        "error_codes": [],
        "vessel_names": [
            "MV LION CITY 01 API REPORTED DISCHARGE WITH CORRELATION CORR-"
        ],
        "voyages": [],
        "terminals": ["PASIR PANJANG TERMINAL 4"],
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
            "stuck": False,
        },
        "text_sample": (
            "EMAIL ALR-861770 | CORR-0001 — DISCHARGE NOT REFLECTED FOR MSKU0000001  "
            "HI OPS  FOR CONTAINER MSKU0000001 ON MV LION CITY 01  API REPORTED DISCHARGE "
            "WITH CORRELATION CORR-0001 AT 2025-10-03 17 20  BUT NO COARRI WAS GENERATED "
            "FOR THIS UNIT  THE LAST EDI ON RECORD IS COPARN REF-COP-0001  08 01   AND T"
        ),
    },
    "confidence": 0.98,
}


if __name__ == "__main__":
    res = process_alert_json(SAMPLE, excel_path=DEFAULT_EXCEL_PATH, max_results=5)
    print("Problem Statement:")
    pprint(res.get("llm_analysis", {}).get("problem_statement"))
    print("SOP:")
    for step in res.get("llm_analysis", {}).get("sop", []):
        print(" -", step)
    print("\nTop matches:")
    for m in res.get("matches", []):
        print(m.get("_match_score"), "-", m.get("subject") or m.get("Subject") or "")
