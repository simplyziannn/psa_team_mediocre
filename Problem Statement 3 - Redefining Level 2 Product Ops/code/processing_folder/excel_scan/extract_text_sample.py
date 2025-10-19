import json
from typing import Any, Dict
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from email_processor.email_processor import process_any
from excel_scanner import check_excel_for_string

def extract_text_sample(result: Dict[str, Any]) -> str:
    """
    Extracts the 'text_sample' from the JSON result produced by email_processor.

    Args:
        result (dict): The JSON object returned by email_processor.

    Returns:
        str: The extracted text sample, or an empty string if not found.
    """
    try:
        return result.get("evidence", {}).get("text_sample", "")
    except Exception:
        return ""


if __name__ == "__main__":
    # directly call email_processor to process email
    result = process_any("Email ALR-861600 | CMAU00000020 - Duplicate Container information received. Hi Jen, Please assist in checking container CMAU00000020. Customer on PORTNET is seeing 2 identical containers information.")

    # Convert to dict if result is a dataclass
    if hasattr(result, "to_dict"):
        result = result.to_dict()

    # Extract and print text sample
    text_sample = extract_text_sample(result)
    print("📝 Text Sample:")
    print(text_sample)

    # passing text sample into excel_scanner.py (check_excel_for_string())
    check_excel_for_string(text_sample)


