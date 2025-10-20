from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

__all__ = ["compile_results"]  # ensures the symbol is exported

def compile_results(
    result: Any,
    json_path: str | Path,
    pdf_output: str | Path,
    save_to_file: bool = True
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """
    Combine DB result, alert_result.json content, and PDF metadata
    into a single structured dict. Optionally writes compiled_output.json
    next to alert_result.json.
    """
    json_path = Path(json_path)

    # Load alert JSON
    try:
        alert_data: Dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        alert_data = {"error": f"Failed to read {json_path}: {e}"}

    # PDF metadata
    pdf_output = str(pdf_output) if pdf_output is not None else ""
    if pdf_output and os.path.exists(pdf_output):
        pdf_meta = {
            "path": str(Path(pdf_output).resolve()),
            "name": Path(pdf_output).name,
            "size_bytes": os.path.getsize(pdf_output),
            "exists": True,
        }
    else:
        pdf_meta = {"path": pdf_output, "exists": False}

    compiled: Dict[str, Any] = {
        "db_result": result,
        "alert_result": alert_data,
        "pdf_info": pdf_meta,
    }

    compiled_path: Optional[Path] = None
    if save_to_file:
        compiled_path = json_path.with_name("compiled_output.json")
        compiled_path.write_text(json.dumps(compiled, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f" Compiled output saved to: {compiled_path}")

    return compiled, compiled_path
