#!/usr/bin/env python3
"""
Minimal test runner for processing.py
- Uses a sample EDI alert
- Prints JSON to stdout
- Saves JSON next to this script
"""

import sys, json
from pathlib import Path
from datetime import datetime
from processing import process_alert, DEFAULT_CASE_LOG, DEFAULT_KB_DOCX, DEFAULT_ESC_PDF,SGT
# --- add to processing.py (near imports) ---
from datetime import datetime, date,timezone

def _json_safe(obj):
    """
    Recursively convert obj into JSON-serializable types.
    - datetime/date -> ISO string
    - numpy/pandas scalars -> native Python types
    - NaT/NaN/None handled to None or str where appropriate
    """
    # Local imports only if available
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import pandas as pd
    except Exception:
        pd = None

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    # pandas Timestamp / NaT
    if pd is not None:
        if isinstance(obj, getattr(pd, "Timestamp", ())):
            return obj.isoformat()
        if obj is getattr(pd, "NaT", None):
            return None

    # numpy types
    if np is not None:
        if isinstance(obj, getattr(np, "integer", ())):
            return int(obj)
        if isinstance(obj, getattr(np, "floating", ())):
            return float(obj)
        if isinstance(obj, getattr(np, "bool_", ())):
            return bool(obj)
        if obj is getattr(np, "nan", None):
            return None

    # anything else that json chokes on -> str fallback
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

SAMPLE_ALERT = [{
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
        "edi_types": ["COARRI", "COPARN"]
    },
    "evidence": {
        "flags": {"duplicate": False, "ack_missing": True, "baplie": False, "coarri": True, "error": False, "stuck": False},
        "text_sample": "EMAIL ALR-861770 | CORR-0001 — DISCHARGE NOT REFLECTED FOR MSKU0000001 ... Last EDI = COPARN REF-COP-0001 ..."
    },
    "confidence": 0.98
},
{ #Strong Case-Log match (should use historical matches; no fallbacks)
  "problem_statement": "EDI REF-COP-0001 stuck without ACK for MSKU0000001", 
  "incident_type": "EDI",
  "variables": {
    "cntr_no": ["MSKU0000001"],
    "message_ref": ["REF-COP-0001"],
    "edi_types": ["COARRI", "COPARN"],
    "terminals": ["PASIR PANJANG TERMINAL 4"],
    "correlation_id": ["CORR-0001"],
    "is_ack_missing_hint": True
  },
  "evidence": {
    "text_sample": "ALR-861770 Email: No COARRI generated; last EDI COPARN REF-COP-0001; correlation CORR-0001."
  },
  "confidence": 0.97
},
{  #No Case-Log match → KB DOCX fallback (Overview sections)
  "problem_statement": "COARRI not received after discharge; verify EDI pipeline health",
  "incident_type": "EDI",
  "variables": {
    "cntr_no": ["TRLU1234567"],
    "message_ref": ["REF-XYZ-9999"],
    "edi_types": ["COARRI"],
    "terminals": ["Tuas Mega Terminal"],
    "is_ack_missing_hint": True
  },
  "evidence": {
    "text_sample": "Container TRLU1234567 discharge event reported; expecting COARRI but none received."
  },
  "confidence": 0.85
},
{ #Still unresolved → Escalation PDF (module routed by incident_type / edi_types)
  "problem_statement": "Unknown EDI error code during COPARN submission",
  "incident_type": "EDI",
  "variables": {
    "cntr_no": ["MSCU7654321"],
    "message_ref": ["REF-NOPE-0000"],
    "edi_types": ["COPARN"],
    "error_codes": ["ERR-UNMAPPED-422"],
    "terminals": ["Brani"]
  },
  "evidence": {
    "text_sample": "Submission to partner returns ERR-UNMAPPED-422; no hits in our knowledge base."
  },
  "confidence": 0.66
},
{ #Solutions present but SOP missing (script should return solutions and leave SOP empty unless KB needed)
  "problem_statement": "EDI duplicate COPARN suspected for CMA booking",
  "incident_type": "EDI",
  "variables": {
    "cntr_no": ["CMAU1111222"],
    "edi_types": ["COPARN"],
    "is_duplicate_hint": True
  },
  "evidence": {
    "text_sample": "Two COPARNs detected for the same unit within 5 minutes. Check dedup logic."
  },
  "confidence": 0.9
},
{ #KB-first flavor (use keywords that you know exist in your KB overviews)
  "problem_statement": "BAPLIE parsing failure for latest vessel call",
  "incident_type": "EDI",
  "variables": {
    "vessel_name": ["MV TESTER"],
    "edi_types": ["BAPLIE"],
    "error_codes": ["ORA-00942"]
  },
  "evidence": {
    "text_sample": "BAPLIE load failed with ORA-00942: table or view does not exist. Need standard recovery steps."
  },
  "confidence": 0.8
},
{ #Minimal alert (should still run; likely KB or Escalation path)
  "problem_statement": "Guan Yibin and Jermaine!! <3",
  "incident_type": "None",
  "variables": {},
  "evidence": {
    "text_sample": "Yibin and Jermaine was caught mating."
  }
},
]

JERMAINE = [{ #Minimal alert (should still run; likely KB or Escalation path)
  "problem_statement": "Guan Yibin and Jermaine!! <3",
  "incident_type": "None",
  "variables": {},
  "evidence": {
    "text_sample": "Yibin and Jermaine was caught mating."
  }
},
{ #Strong Case-Log match (should use historical matches; no fallbacks)
  "problem_statement": "EDI REF-COP-0001 stuck without ACK for MSKU0000001", 
  "incident_type": "EDI",
  "variables": {
    "cntr_no": ["MSKU0000001"],
    "message_ref": ["REF-COP-0001"],
    "edi_types": ["COARRI", "COPARN"],
    "terminals": ["PASIR PANJANG TERMINAL 4"],
    "correlation_id": ["CORR-0001"],
    "is_ack_missing_hint": True
  },
  "evidence": {
    "text_sample": "ALR-861770 Email: No COARRI generated; last EDI COPARN REF-COP-0001; correlation CORR-0001."
  },
  "confidence": 0.97
},
{ #Still unresolved → Escalation PDF (module routed by incident_type / edi_types)
  "problem_statement": "Unknown EDI error code during COPARN submission",
  "incident_type": "EDI",
  "variables": {
    "cntr_no": ["MSCU7654321"],
    "message_ref": ["REF-NOPE-0000"],
    "edi_types": ["COPARN"],
    "error_codes": ["ERR-UNMAPPED-422"],
    "terminals": ["Brani"]
  },
  "evidence": {
    "text_sample": "Submission to partner returns ERR-UNMAPPED-422; no hits in our knowledge base."
  },
  "confidence": 0.66
}
]


def _save_here(obj: dict, base_name="test_alert_result") -> Path:
    here = Path(__file__).resolve().parent
    ts = datetime.now(SGT).strftime("%Y%m%dT%H%M%SZ")
    out = here /"JSON OUTPUT"/ f"{base_name}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out

def main(json_input):  
    case_log = Path(DEFAULT_CASE_LOG)
    kb_docx  = Path(DEFAULT_KB_DOCX)
    esc_pdf  = Path(DEFAULT_ESC_PDF)
    args = sys.argv
    if "--excel" in args: case_log = Path(args[args.index("--excel")+1])
    if "--kb" in args:    kb_docx  = Path(args[args.index("--kb")+1])
    if "--pdf" in args:   esc_pdf  = Path(args[args.index("--pdf")+1])

    res = process_alert(json_input, case_log, kb_docx, esc_pdf, max_results=10)
    safe = _json_safe(res)
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    saved = _save_here(safe, base_name="alert_result")

    sys.stderr.write(f"\nSaved JSON: {saved}\n")

    
if __name__ == "__main__":
    # Optional overrides:
    #for json_input in SAMPLE_ALERT:
    for json_input in JERMAINE:
        main(json_input)
