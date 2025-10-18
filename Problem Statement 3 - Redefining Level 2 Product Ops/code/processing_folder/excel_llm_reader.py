# excel_llm_reader.py
from __future__ import annotations
import os, io, json, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

# --- 0) LLM adapter ----------------------------------------------------------
# If you already have a helper (e.g., `from LLM_folder.call_openai_basic import ask_gpt5`),
# replace ask_llm() with a thin wrapper that calls your helper and returns the string content.

def ask_llm(messages: List[Dict[str, str]]) -> str:
    """
    Minimal adapter. Replace this with your own client.
    Must return the assistant's text content.
    """
    # Example: using requests to your Azure/OpenAI gateway (pseudo):
    #
    # import requests
    # url = os.getenv("LLM_URL")  # e.g., https://.../chat/completions
    # headers = {"Content-Type": "application/json", "Ocp-Apim-Subscription-Key": os.getenv("APIM_KEY","")}
    # resp = requests.post(url, json={"messages": messages, "max_completion_tokens": 2048, "response_format": {"type": "text"}}, headers=headers, timeout=60)
    # resp.raise_for_status()
    # return resp.json()["choices"][0]["message"]["content"]
    raise NotImplementedError("Plug in your LLM client in ask_llm().")


# --- 1) XLSX -> compact text snapshot ---------------------------------------
@dataclass
class SnapshotConfig:
    max_sheets: int = 6
    max_rows_per_sheet: int = 40
    max_cols_per_sheet: int = 20
    drop_empty_rows: bool = True
    strip_whitespace: bool = True

def _clean_df(df: pd.DataFrame, cfg: SnapshotConfig) -> pd.DataFrame:
    df = df.copy()
    # Convert everything to string for consistent search & display
    df = df.applymap(lambda x: "" if pd.isna(x) else str(x))
    if cfg.strip_whitespace:
        df = df.applymap(lambda s: s.strip())
    if cfg.drop_empty_rows:
        df = df[~(df.apply(lambda r: "".join(r.values.astype(str)).strip() == "", axis=1))]
    return df

def _df_preview_markdown(df: pd.DataFrame, cfg: SnapshotConfig) -> str:
    # Limit size for tokens
    df_small = df.iloc[:cfg.max_rows_per_sheet, :cfg.max_cols_per_sheet]
    # Use a lightweight markdown-esque table
    buf = io.StringIO()
    headers = ["|"] + [c if c is not None else "" for c in df_small.columns] + ["|"]
    sep = ["|"] + ["---"] * len(df_small.columns) + ["|"]
    print(" ".join(headers), file=buf)
    print(" ".join(sep), file=buf)
    for _, row in df_small.iterrows():
        cells = ["|"] + [row[c] for c in df_small.columns] + ["|"]
        print(" ".join(cells), file=buf)
    return buf.getvalue()

def xlsx_to_snapshot_text(path: str, cfg: SnapshotConfig = SnapshotConfig()) -> str:
    """
    Create a compact, token-friendly text snapshot of all sheets (truncated).
    """
    xl = pd.ExcelFile(path)
    sheets = xl.sheet_names[:cfg.max_sheets]
    parts = []
    for s in sheets:
        try:
            df = xl.parse(s, dtype=str)  # read as strings
            df = _clean_df(df, cfg)
            md = _df_preview_markdown(df, cfg)
            parts.append(f"# Sheet: {s}\n{md}")
        except Exception as e:
            parts.append(f"# Sheet: {s}\n<error reading sheet: {e}>")
    return "\n\n".join(parts)


# --- 2) Local verifier (ground truth) ----------------------------------------
@dataclass
class Hit:
    sheet: str
    row_index: int
    col_name: str
    value: str

def local_find_string_in_xlsx(path: str, needle: str, case_sensitive: bool = False,
                              cfg: SnapshotConfig = SnapshotConfig()) -> List[Hit]:
    """
    Fast local scan over the FULL sheets (not truncated) for verification.
    Returns all matches with sheet, row, column, and value.
    """
    xl = pd.ExcelFile(path)
    hits: List[Hit] = []
    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.compile(re.escape(needle), flags=flags)

    for s in xl.sheet_names:
        df = xl.parse(s, dtype=str)
        df = df.applymap(lambda x: "" if pd.isna(x) else str(x))
        for r_i, row in df.iterrows():
            for c in df.columns:
                val = row[c]
                if pat.search(val or ""):
                    hits.append(Hit(sheet=s, row_index=r_i, col_name=str(c), value=val))
    return hits


# --- 3) LLM Q&A over snapshot ------------------------------------------------
def ask_about_xlsx(path: str, question: str, cfg: SnapshotConfig = SnapshotConfig(),
                   json_mode: bool = True) -> Dict[str, Any]:
    """
    Sends a truncated snapshot to the LLM and asks your question.
    If json_mode=True, we instruct the model to answer in strict JSON so you can parse it.
    """
    snapshot = xlsx_to_snapshot_text(path, cfg)

    system = (
        "You are a precise data extraction assistant. You are given a truncated markdown snapshot "
        "of an Excel workbook (some rows/cols may be omitted). Work ONLY with the given snapshot."
    )
    if json_mode:
        user = f"""You are given a snapshot of an XLSX file:

---BEGIN SNAPSHOT---
{snapshot}
---END SNAPSHOT---

Task: Answer the following question ONLY using the snapshot above.
Question: {question}

Respond ONLY in strict JSON with this schema:
{{
  "answer": "yes" | "no" | "unknown",
  "explanation": "short reason",
  "evidence": [{{"sheet": str, "row_index": int, "col": str, "value": str}}]
}}

Rules:
- "yes" only if you can point to at least one matching cell in the snapshot.
- "no" only if you have high confidence it's absent in the SHOWN snapshot.
- "unknown" if the snapshot might be truncated in ways that hide the answer."""
    else:
        user = f"SNAPSHOT:\n{snapshot}\n\nQuestion: {question}\nAnswer succinctly."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    content = ask_llm(messages)  # <-- Plug in your client
    try:
        return json.loads(content)
    except Exception:
        return {"answer": "unknown", "explanation": "LLM did not return valid JSON.", "raw": content, "evidence": []}


# --- 4) One-shot 'is ... in the xlsx?' probe (LLM + local cross-check) ------
def probe_contains(path: str, phrase: str, cfg: SnapshotConfig = SnapshotConfig(),
                   do_local_verify: bool = True) -> Dict[str, Any]:
    """
    Ask the LLM if 'phrase' is in the XLSX snapshot AND (optionally) verify locally on the full file.
    This lets you manually check whether the LLM actually saw it.
    """
    q = f'Is the exact phrase "{phrase}" present in any cell of the shown snapshot? ' \
        f'If yes, show where (sheet, row, col, value).'
    llm_result = ask_about_xlsx(path, q, cfg=cfg, json_mode=True)

    out: Dict[str, Any] = {"phrase": phrase, "llm": llm_result}
    if do_local_verify:
        local_hits = local_find_string_in_xlsx(path, phrase, case_sensitive=False)
        out["local"] = {
            "found": len(local_hits) > 0,
            "count": len(local_hits),
            "evidence": [hit.__dict__ for hit in local_hits[:50]]  # avoid huge dumps
        }
    return out
