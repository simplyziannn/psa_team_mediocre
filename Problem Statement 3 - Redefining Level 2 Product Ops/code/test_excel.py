import sys
from pprint import pprint

from excel_processing import (
    process_email,
    find_matching_cases,
    DEFAULT_EXCEL_PATH,
)


def run_online():
    sample_email = {
        "subject": "Customer ABC: Frequent 500 errors on Vessel Registry",
        "body": (
            "Hi team, we are seeing ERR-5023 and HTTP 500 on "
            "vessel_registry_service. Prior case: VR-1042."
        ),
    }
    print("== Running online test ==")
    result = process_email(sample_email, excel_path=DEFAULT_EXCEL_PATH)
    print("Structured email:")
    pprint(result["structured_email"])
    print("\\nProblem Statement:")
    pprint(result.get("llm_analysis", {}).get("problem_statement"))
    print("SOP:")
    for step in result.get("llm_analysis", {}).get("sop", []):
        print(" -", step)
    print("\nTop matches:")
    for r in result["matches"]:
        print(r.get("_match_score"), "-", r.get("subject") or r.get("Subject") or "")


def run_offline():
    entities = {
        "subject": "Frequent 500 errors on Vessel Registry",
        "customer": "Customer ABC",
        "product": "Vessel Registry",
        "environment": None,
        "error_codes": ["ERR-5023", "500"],
        "services": ["vessel_registry_service"],
        "keywords": ["500", "ERR-5023", "vessel", "registry", "service"],
        "case_ids": ["VR-1042"],
        "probable_issue": None,
        "suspected_component": None,
        "summary": "Customer sees repeated 500s",
    }
    print("== Running offline test ==")
    matches = find_matching_cases(entities, excel_path=DEFAULT_EXCEL_PATH)
    print(f"Excel: {DEFAULT_EXCEL_PATH}")
    print("Top matches:")
    for r in matches:
        print(r.get("_match_score"), "-", r.get("subject") or r.get("Subject") or "")


if __name__ == "__main__":
    try:
        if "--offline" in sys.argv:
            run_offline()
        else:
            run_online()
    except Exception as e:
        print(f"Test failed: {e}")

