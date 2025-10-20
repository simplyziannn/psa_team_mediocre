import sys, os, json, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing_folder.doc_scan import docx_scanner
from pathlib import Path
from docx import Document
from LLM_folder.call_openai_basic import ask_gpt5

# --------------------------------------------------------------------
# Path to Knowledge Base
# --------------------------------------------------------------------
KNOWLEDGE_BASE_DOCX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "Info",
    "Knowledge Base.docx"
)

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

def _coerce_llm_text(result) -> str:
    """Handle ask_gpt5 returning either str or dict (OpenAI-like)."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        try:
            choices = result.get("choices") or []
            if choices:
                ch0 = choices[0]
                msg = ch0.get("message") or {}
                content = msg.get("content")
                if content:
                    return str(content)
                if "text" in ch0 and ch0["text"]:
                    return str(ch0["text"])
        except Exception:
            pass
        return json.dumps(result)
    return str(result)


def _extract_json_block(text: str) -> str:
    """Extract the first JSON object from text (handles ```json fences)."""
    if not text:
        raise ValueError("Empty LLM text")
    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # plain {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    raise ValueError("No JSON object found in LLM text")


def _escape_ctrl_chars_in_strings(s: str) -> str:
    """Escape \n, \r, \t inside quoted strings to avoid JSONDecodeError."""
    out = []
    in_str = False
    esc = False
    quote = None
    for ch in s:
        if esc:
            out.append(ch)
            esc = False
            continue
        if ch == '\\':
            out.append(ch)
            esc = True
            continue
        if in_str:
            if ch == quote:
                in_str = False
                out.append(ch)
            elif ch == '\n':
                out.extend(['\\', 'n'])
            elif ch == '\r':
                out.extend(['\\', 'r'])
            elif ch == '\t':
                out.extend(['\\', 't'])
            else:
                out.append(ch)
        else:
            if ch in ('"', "'"):
                in_str = True
                quote = ch
            out.append(ch)
    return ''.join(out)


def _convert_sop_to_list(data: dict,llm=False) -> dict:
    """Convert multi-line SOP string to list of lines (SOP_lines)."""
    sop = data.get("SOP", "")
    if isinstance(sop, str):
        lines = [ln.rstrip() for ln in sop.splitlines() if ln.strip()]
        if llm:
            data["SOP"] = lines
    return data


def _save_json_pretty(data: dict, filename: str = "alert_result.json") -> str:
    """Save JSON data (with SOP_lines) into JSON OUTPUT folder."""
    out_dir = Path(__file__).parent / "JSON OUTPUT"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f" JSON saved (pretty) to: {out_path.resolve()}")
    print("\n Preview:\n" + json.dumps(data, indent=2, ensure_ascii=False))
    return str(out_path.resolve())


# --------------------------------------------------------------------
# Main Logic
# --------------------------------------------------------------------
 
def docx_read_and_save(tuple_parameter,filepath=KNOWLEDGE_BASE_DOCX) -> str:
    """Read DOCX, query LLM, extract JSON, convert SOP to list, save JSON."""
    problem_statement,solution,SOP,excel_matched = tuple_parameter
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f" File not found: {filepath}")

    # 1️⃣ Read DOCX
    doc = Document(path)
    text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text and p.text.strip())
    if not text.strip():
        raise ValueError(f"️ File is empty or unreadable: {filepath}")
    excel_matched = False
    # 2️⃣ Build LLM prompt
    #make more conditions
    zero_match = problem_statement == solution == SOP
    if excel_matched==False or zero_match:
        #problem_match(problem_statement,text)
        problem_statement = "Callback delivery failures for NYKU8964499. Webhook sender faced 401 due to peer TLS renegotiation. Queue grew and breaker opened. Affected endpoint '/edi/upload'."
        json_results = docx_scanner.main(problem_statement,KNOWLEDGE_BASE_DOCX)
        return converting_json_file(json_results)
    elif SOP!='0':
        #SOP_match(SOP,text)
        SOP = "EDI: Spike in DLQ messages after routine maintenance; consumer group lag increased across EDI topic"
        return converting_json_file(docx_scanner.main(SOP,KNOWLEDGE_BASE_DOCX))
    elif SOP == '0' and solution == '0' and problem_statement!='0':
        #look for problem statement when no SOP under overview section
        return converting_json_file(docx_scanner.main(problem_statement,KNOWLEDGE_BASE_DOCX))
    elif SOP =='0' and solution != '0' and problem_statement != '0': 
        print(solution)
        print("NO SOP Found, only solutions")
        return converting_json_file(solution)


def problem_match(problem,text):
        problem = (
            "We are attempting to create a container range from CONTAINER_ID to CONTAINER_ID, "
            "but encountered an error: 'Overlapping container range(s) found.' However, the specified range could not "
            "be located. A query for the range from CONTAINER_ID to CONTAINER_ID returned a different range. "
            "The concern is that the range isn't visible in the system, despite being created by the relevant process. "
            "A search on our end also returned a different range."
        )

        preview = text[:1500]
        prompt = (
            "You are verifying a DOCX reader pipeline.\n\n"
            f"Current Problem: {problem}\n\n"
            'Look for similar problems under "Overview" and respond ONLY as valid JSON with exactly these keys:\n'
            "{\n"
            '  \"CNTR\": \"cntr message\",\n'
            '  \"SOP\": \"SOP procedure + verification in proper indentation\"\n'
            "}\n"
            "Do not include any extra text or markdown.\n\n"
            f"Document content (first 1500 chars):\n{preview}"
        )

        # 3️⃣ Query LLM
        print(" Sending preview to LLM...")
        raw = ask_gpt5(prompt)
        print(" LLM responded")

        # 4️⃣ Coerce & extract JSON
        converting_json_file(raw)

def converting_json_file(raw,llm=False):
        print(" LLM responded")

        # 4️⃣ Coerce & extract JSON
        text_out = _coerce_llm_text(raw)
        try:
            json_str = _extract_json_block(text_out)
        except ValueError:
            json_str = text_out.strip()

        # 5️⃣ Parse JSON safely
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            fixed = _escape_ctrl_chars_in_strings(json_str)
            parsed = json.loads(fixed)

        # 6️⃣ Convert SOP → list
        parsed = _convert_sop_to_list(parsed,llm)

        # 7️⃣ Save and return path
        return _save_json_pretty(parsed, "alert_result.json")
    
def SOP_match(CNTR,text):
    if CNTR is None:
        print("CNTR not avaiable.")
        return
    CNTR = (
        "EDI: Spike in DLQ messages after routine maintenance; consumer group lag increased across EDI topic"
    )

    preview = text[:1500]
    prompt = (
        "You are verifying a DOCX reader pipeline.\n\n"
        f"Current Problem: {CNTR}\n\n"
        'Look for similar CNTR header under "Overview" and respond ONLY as valid JSON with exactly these keys:\n'
        "{\n"
        '  \"CNTR\": \"cntr message\",\n'
        '  \"SOP\": \"SOP procedure + verification in proper indentation\"\n'
        "}\n"
        "Do not include any extra text or markdown.\n\n"
        f"Document content (first 1500 chars):\n{preview}"
    )

    # 3️⃣ Query LLM
    print(" Sending preview to LLM...")
    raw = ask_gpt5(prompt)
    print(" LLM responded")

    # 4️⃣ Coerce & extract JSON
    text_out = _coerce_llm_text(raw)
    try:
        json_str = _extract_json_block(text_out)
    except ValueError:
        json_str = text_out.strip()

    # 5️⃣ Parse JSON safely
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        fixed = _escape_ctrl_chars_in_strings(json_str)
        parsed = json.loads(fixed)

    # 6️⃣ Convert SOP → list
    parsed = _convert_sop_to_list(parsed)

    # 7️⃣ Save and return path
    return _save_json_pretty(parsed, "alert_result.json")
    return
    
def main(test_tuple):
    saved_path = docx_read_and_save(test_tuple)
    print(f"\n Done. JSON saved at:\n{saved_path}")
    return saved_path


if __name__ == "__main__":
    test_tuple = ("GYB","Jermaine","GYB and Jermaine",True)
    main(test_tuple)
