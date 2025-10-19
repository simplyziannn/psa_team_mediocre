import os, sys, io, re
from typing import Any, Dict, Optional, Tuple
from contextlib import redirect_stdout
from openpyxl import load_workbook

# ---- Ensure we can import from project root ----
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# ---- Imports from your project ----
from email_processor.email_processor import process_any
from processing_folder.excel_scan.excel_scanner import check_excel_for_string
from processing_folder.excel_scan.extract_text_sample import extract_text_sample


# --------------------------------------------------------------------
# Function: get_highest_confidence_from_printed
# --------------------------------------------------------------------
def get_highest_confidence_from_printed(printed_output: str, debug: bool = True) -> Optional[Dict[str, Any]]:
    """
    Extracts the highest-confidence match from the printed scanner output.

    Logic:
      - Find the first occurrence of '[Sheet1 r### c###]' after '✅ Found matches:'
      - Extract row and col values.
      - Since the scanner output is already sorted by confidence, this is the top match.
    Returns:
      {"sheet": str, "row": int, "col": int, "score": float} or None.
    """
    if "✅ Found matches:" not in printed_output:
        if debug:
            print("❌ No 'Found matches' section in output.")
        return None

    after_marker = printed_output.split("✅ Found matches:")[-1].strip()

    _SCORE_RE = re.compile(r"(?P<score>\d+\.\d+)")
    _LOCATOR_RE = re.compile(r"\[(?P<sheet>[A-Za-z0-9_ ]+)\s*r(?P<row>\d+)\s*c(?P<col>\d+)\]")

    mloc = _LOCATOR_RE.search(after_marker)
    mscore = _SCORE_RE.search(after_marker)

    if not mloc:
        if debug:
            print("❌ Could not locate any [Sheet r c] pattern.")
        return None

    sheet = mloc.group("sheet").strip()
    row = int(mloc.group("row"))
    col = int(mloc.group("col"))
    score = float(mscore.group("score")) if mscore else None

    result = {"sheet": sheet, "row": row, "col": col, "score": score}

    if debug:
        print(f"🏆 Top match -> Sheet: {sheet}, Row: {row}, Col: {col}, Score: {score}")

    return result

def read_fixed_columns_from_excel(xlsx_path: str, sheet: str, row: int, debug: bool = True) -> Tuple[Any, Any, Any, bool]:
    """
    Reads values from the specified Excel sheet and row, shifted by one column:
      now reads columns 6, 7, and 8 (F, G, H).
    Returns:
        (col6, col7, col8, match_status)
        where match_status = True if any non-empty cell exists, otherwise False.
    """

    wb = load_workbook(xlsx_path, data_only=True)
    if sheet not in wb.sheetnames:
        if debug:
            print(f"⚠️ Sheet '{sheet}' not found, using first sheet instead.")
        ws = wb.active
    else:
        ws = wb[sheet]

    excel_row = row + 1 if row >= 0 else row

    # Shifted columns → F (6), G (7), H (8)
    values = []
    for col in [6, 7, 8]:
        cell_value = ws.cell(row=excel_row, column=col).value
        values.append(cell_value if (cell_value is not None and str(cell_value).strip() != "") else 0)

    match_status = any(v != 0 for v in values)

    if debug:
        coords = [ws.cell(row=excel_row, column=col).coordinate for col in [6, 7, 8]]
        print(f"📘 Reading from: {xlsx_path}")
        print(f"📄 Sheet: {ws.title}, Row: {excel_row}, Columns: F/G/H")
        print(f"🧭 Cells: {coords}")
        print(f"📖 Values: {values}")
        print(f"✅ Match found: {match_status}")

    return (*values, match_status)

# --------------------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Process the email and get text to search
    result = process_any(
        "Email ALR-861600 | CMAU00000020 - Duplicate Container information received. "
        "Hi Jen, Please assist in checking container CMAU00000020. Customer on PORTNET is seeing 2 identical containers information."
    )
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    text_sample = extract_text_sample(result)

    # 2) Run the Excel scanner and capture its printed output
    buf = io.StringIO()
    with redirect_stdout(buf):
        check_excel_for_string(text_sample)
    printed_output = buf.getvalue()

    # 3) Parse the top match (already highest confidence)
    top = get_highest_confidence_from_printed(printed_output, debug=True)
    if not top:
        print((0, 0, 0, False))
        sys.exit(0)

    # 4) Read F/G/H on that row (shifted columns 6,7,8)
    excel_path = "/Users/zian/Documents/PSA Hackathon/PSA_Mediocre/Problem Statement 3 - Redefining Level 2 Product Ops/code/processing_folder/excel_scan/Case Log.xlsx"
    col6, col7, col8, matched = read_fixed_columns_from_excel(
        excel_path, sheet=top["sheet"], row=top["row"], debug=True
    )

    print("\n🎯 Final result:", (col6, col7, col8, matched))
