"""Extract seed INSERTs from Info/Database/db.sql into JSON files.

Produces JSON files for: vessels, containers, edi_messages, api_events

Assumptions/limitations:
- Parses simple `INSERT INTO table (col, col, ...) VALUES (v1, v2, ...), (v3, v4, ...);` blocks.
- Handles NULL, numeric, string (single-quoted) and JSON_OBJECT(...) tokens by preserving as strings for now.
- This script is tailored to this repository's `db.sql` seed format; it doesn't fully implement an SQL parser.
"""
import re
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]  # repo root: .../Problem Statement 3 - Redefining Level 2 Product Ops/code/db_folder/.. -> code/
SQL_PATH = BASE / 'Info' / 'Database' / 'db.sql'
OUT_DIR = Path(__file__).resolve().parent / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = {
    'vessel': 'vessels.json',
    'container': 'containers.json',
    'edi_message': 'edi_messages.json',
    'api_event': 'api_events.json',
}

# Regex to locate INSERT INTO ... (cols) VALUES ...; capturing table, cols and values block
INSERT_RE = re.compile(r'''INSERT\s+INTO\s+([`"]?)(?P<table>\w+)\1\s*\((?P<cols>[^)]+)\)\s*VALUES\s*(?P<values>.+?);''', re.IGNORECASE | re.DOTALL)

# Split columns by comma
def split_cols(col_text):
    return [c.strip().strip('`"') for c in col_text.split(',')]

# Split top-level value tuples separated by '),(' --- naive but OK for our seed file
def split_value_tuples(values_text):
    """Split the VALUES (...) , (...) block into individual tuple strings.

    This version tracks parentheses and single-quote quoting so commas inside
    strings don't split tuples.
    """
    txt = values_text.strip()
    tuples = []
    current = ''
    depth = 0
    in_quote = False
    i = 0
    while i < len(txt):
        ch = txt[i]
        current += ch
        if ch == "'":
            # handle escaped '' by skipping the next quote
            if in_quote and i + 1 < len(txt) and txt[i+1] == "'":
                # add the escaped quote and advance
                current += "'"
                i += 1
            else:
                in_quote = not in_quote
        elif not in_quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    # finished a tuple
                    tuples.append(current.strip())
                    current = ''
                    # skip any following whitespace and comma
                    j = i + 1
                    while j < len(txt) and txt[j].isspace():
                        j += 1
                    if j < len(txt) and txt[j] == ',':
                        i = j  # will be incremented at loop end
        i += 1
    # filter out empty
    return [t for t in tuples if t]

# Parse a single tuple like: (1, 'ABC', NULL, '2025-10-03 08:01', JSON_OBJECT('bay','12'))
VAL_RE = re.compile(r"'([^']*(?:''[^']*)*)'|NULL|([+-]?\d+\.?\d*)|JSON_OBJECT\([^)]*\)|NOW\(\)|NOW\(\) \+ INTERVAL [^,\)]+|[^,]+", re.IGNORECASE)

def parse_tuple(tuple_text):
    # Strip surrounding parentheses
    t = tuple_text.strip()
    if t.startswith('(') and t.endswith(')'):
        t = t[1:-1]
    parts = []
    i = 0
    length = len(t)
    while i < length:
        ch = t[i]
        if ch.isspace():
            i += 1
            continue
        if t[i] == "'":
            # quoted string, handle escaped quotes by doubling
            i += 1
            s = ''
            while i < length:
                if t[i] == "'":
                    if i + 1 < length and t[i+1] == "'":
                        s += "'"
                        i += 2
                        continue
                    else:
                        i += 1
                        break
                else:
                    s += t[i]
                    i += 1
            parts.append(s)
            # skip optional whitespace and comma
            while i < length and t[i].isspace():
                i += 1
            if i < length and t[i] == ',':
                i += 1
        elif t[i:i+11].upper() == 'JSON_OBJECT':
            # capture until matching )
            start = i
            depth = 0
            while i < length:
                if t[i] == '(':
                    depth += 1
                elif t[i] == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            parts.append(t[start:i].strip())
            # skip comma
            while i < length and (t[i].isspace() or t[i] == ','):
                i += 1
        elif t[i:i+3].upper() == 'NOW':
            # NOW() or NOW() + INTERVAL 1 SECOND
            start = i
            while i < length and t[i] != ',':
                i += 1
            parts.append(t[start:i].strip())
            if i < length and t[i] == ',':
                i += 1
        else:
            # unquoted token (number, NULL, etc.) read until comma
            start = i
            while i < length and t[i] != ',':
                i += 1
            token = t[start:i].strip()
            if token.upper() == 'NULL':
                parts.append(None)
            else:
                # try number
                try:
                    if '.' in token:
                        parts.append(float(token))
                    else:
                        parts.append(int(token))
                except Exception:
                    parts.append(token)
            if i < length and t[i] == ',':
                i += 1
    return parts


def process_sql(sql_text):
    results = {k: [] for k in TARGETS.keys()}
    for m in INSERT_RE.finditer(sql_text):
        table = m.group('table')
        cols = split_cols(m.group('cols'))
        values = m.group('values')
        tuples = split_value_tuples(values)
        for tup in tuples:
            vals = parse_tuple(tup)
            # align length
            row = {}
            for i, col in enumerate(cols):
                v = vals[i] if i < len(vals) else None
                row[col] = v
            if table in results:
                results[table].append(row)
    return results


def main():
    if not SQL_PATH.exists():
        print(f"Could not find {SQL_PATH}. Run this script from the repo with the db.sql present.")
        return
    sql_text = SQL_PATH.read_text(encoding='utf-8')
    extracted = process_sql(sql_text)
    for table, rows in extracted.items():
        out_name = TARGETS.get(table)
        if not out_name:
            continue
        out_path = OUT_DIR / out_name
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False, default=str)
        print(f"Wrote {len(rows)} rows to {out_path}")

if __name__ == '__main__':
    main()
