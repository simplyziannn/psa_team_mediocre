"""Query generated seed JSON files and print matching records as JSON.

Usage examples:
python code/db_folder/query_seed.py --vessel_name "MV Lion City 01"
python code/db_folder/query_seed.py --cntr_no MSCU0000006
python code/db_folder/query_seed.py --correlation_id corr-0006
python code/db_folder/query_seed.py --message_ref REF-COP-0001

Behavior:
- Loads JSON files from `code/db_folder/output/` (vessels.json, containers.json, edi_messages.json, api_events.json)
- Searches for the provided key and prints matches as JSON lines (one JSON object per match)
- Supports case-insensitive substring matching for string fields
"""
import argparse
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent / 'output'
FILES = {
    'vessels': OUT / 'vessels.json',
    'containers': OUT / 'containers.json',
    'edi_messages': OUT / 'edi_messages.json',
    'api_events': OUT / 'api_events.json',
}


def load_json(path):
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def find_by_key(data_list, key, value, case_insensitive=True):
    matches = []
    for obj in data_list:
        if key not in obj:
            continue
        v = obj[key]
        if v is None:
            continue
        if isinstance(v, (int, float)):
            if str(v) == str(value):
                matches.append(obj)
        else:
            if case_insensitive:
                if str(value).lower() in str(v).lower():
                    matches.append(obj)
            else:
                if str(value) == str(v):
                    matches.append(obj)
    return matches


def main():
    parser = argparse.ArgumentParser()
    # Allow multiple filters at once; we will return all matches across provided keys
    parser.add_argument('--vessel_name')
    parser.add_argument('--cntr_no')
    parser.add_argument('--correlation_id')
    parser.add_argument('--message_ref')
    parser.add_argument('--first', action='store_true', help='Print only the first match')
    args = parser.parse_args()

    if not (args.vessel_name or args.cntr_no or args.correlation_id or args.message_ref):
        parser.error('At least one of --vessel_name, --cntr_no, --correlation_id or --message_ref must be provided')

    # Load files
    vessels = load_json(FILES['vessels'])
    containers = load_json(FILES['containers'])
    edi = load_json(FILES['edi_messages'])
    api = load_json(FILES['api_events'])

    results = []
    # Collect matches from each requested dataset
    if args.vessel_name:
        results += find_by_key(vessels, 'vessel_name', args.vessel_name)
    if args.cntr_no:
        results += find_by_key(containers, 'cntr_no', args.cntr_no)
    if args.correlation_id:
        results += find_by_key(api, 'correlation_id', args.correlation_id)
    if args.message_ref:
        results += find_by_key(edi, 'message_ref', args.message_ref)

    # Deduplicate results by JSON representation (stable)
    seen = set()
    unique_results = []
    for r in results:
        key = json.dumps(r, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    if not unique_results:
        # Print a JSON error object for downstream LLM consumption
        err = {"error": "No matches found for provided query parameters"}
        print(json.dumps(err, indent=2, ensure_ascii=False))
        sys.exit(1)

    if args.first:
        print(json.dumps(unique_results[0], indent=2, ensure_ascii=False))
        return

    # Print all matches
    print(json.dumps(unique_results, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
