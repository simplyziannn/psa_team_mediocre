import os
import json
from typing import Optional, Dict, Any

def extract_module_from_json() -> Optional[str]:
    """
    Reads 'alert_result.json' from the JSON OUTPUT folder (one level up from this script)
    and extracts the 'module' field safely.
    Works on all PCs and OS (no hardcoded paths).
    """
    try:
        # Step 1: Get directory of the current script (pdf_scan/)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Step 2: Go up one level to reach processing_folder/
        parent_dir = os.path.dirname(current_dir)

        # Step 3: Build full path to JSON OUTPUT/alert_result.json
        json_path = os.path.join(parent_dir, "JSON OUTPUT", "alert_result.json")

        # Step 4: Open and parse the JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        # Step 5: Extract 'module' value
        module_value = data.get("owner", {}).get("module")
        return module_value

    except FileNotFoundError:
        print(" 'alert_result.json' not found in JSON OUTPUT folder.")
    except json.JSONDecodeError:
        print(" Invalid JSON format in 'alert_result.json'.")
    except Exception as e:
        print(f"️ Unexpected error: {e}")

    return None


# Example usage
if __name__ == "__main__":
    module = extract_module_from_json()
    print("Module:", module)
