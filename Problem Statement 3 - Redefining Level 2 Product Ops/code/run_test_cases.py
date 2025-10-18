import os
import re
from typing import List, Dict

from excel_processing import process_email, DEFAULT_EXCEL_PATH


HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(os.path.dirname(HERE), "Info", "Test Cases.pdf")


def extract_pdf_text(path: str) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception as e:
        raise RuntimeError(
            "PyPDF2 is required to read the PDF. Install it with: pip install PyPDF2"
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n\n".join(texts)


def split_cases(text: str) -> List[Dict[str, str]]:
    # Heuristics: split on headings like "Email:", "SMS:", "Call:". Keep the label as type.
    # If no headings, fallback to large paragraphs.
    parts: List[Dict[str, str]] = []
    pattern = re.compile(r"(?mi)^(Email|SMS|Call)\s*:?\s*")
    idxs = [(m.start(), m.group(1)) for m in pattern.finditer(text)]

    if not idxs:
        # Fallback: split by multiple blank lines
        chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
        for c in chunks:
            parts.append({"type": "Unknown", "text": c})
        return parts

    # Chunk from each heading to the next
    for i, (start, label) in enumerate(idxs):
        end = idxs[i + 1][0] if i + 1 < len(idxs) else len(text)
        chunk = text[start:end].strip()
        # Remove the heading itself from the text
        chunk_body = pattern.sub("", chunk, count=1).strip()
        parts.append({"type": label, "text": chunk_body})
    return parts


def infer_subject_and_body(case: Dict[str, str]) -> Dict[str, str]:
    t = case.get("text", "").strip()
    # Try explicit Subject:
    m = re.search(r"(?mi)^subject\s*:\s*(.+)$", t)
    if m:
        subject = m.group(1).strip()
        body = re.sub(r"(?mi)^subject\s*:\s*.+$", "", t).strip()
    else:
        # Use first line as subject for non-email too
        first_line = (t.splitlines() or [""])[0].strip()
        subject = (first_line[:120]).strip()
        body = t
    return {"subject": subject, "body": body}


def main():
    print(f"PDF: {PDF_PATH}")
    print(f"Excel: {DEFAULT_EXCEL_PATH}")
    text = extract_pdf_text(PDF_PATH)
    cases = split_cases(text)
    print(f"Found cases: {len(cases)}")

    for i, c in enumerate(cases, 1):
        sb = infer_subject_and_body(c)
        print("\n== Case", i, f"({c['type']}) ==")
        print("Subject:", sb["subject"])    
        try:
            res = process_email(sb, excel_path=DEFAULT_EXCEL_PATH, max_results=5)
        except Exception as e:
            print("Error processing case:", e)
            continue
        se = res.get("structured_email", {})
        print("Entities:")
        print(" - product:", se.get("product"))
        print(" - error_codes:", se.get("error_codes"))
        print(" - services:", se.get("services"))
        print(" - case_ids:", se.get("case_ids"))
        print("Top matches:")
        for r in res.get("matches", []):
            subj = r.get("subject") or r.get("Subject") or ""
            print(r.get("_match_score"), '-', subj)


if __name__ == "__main__":
    main()

