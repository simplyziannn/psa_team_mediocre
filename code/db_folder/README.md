# db_folder extractor

This small utility extracts seed INSERT statements from `Info/Database/db.sql` and writes JSON files for ingestion by an LLM or downstream tools.

Usage

1. From the repository root run:

```bash
python code/db_folder/extract_seed_json.py
```

2. Outputs are written to `code/db_folder/output/`:
- `vessels.json`
- `containers.json`
- `edi_messages.json`
- `api_events.json`

Notes and limitations
- This script is a pragmatic parser for the seed file format found in `db.sql`. It doesn't fully parse arbitrary SQL and may fail on more complex INSERT forms.
- JSON-typed SQL functions like `JSON_OBJECT(...)` are preserved as strings; you can further parse them if needed.
- Date and NOW() expressions are stringified.

If you'd like, I can:
- Run the script now and show the produced JSON
- Improve JSON_OBJECT handling to produce proper JSON objects
- Add a tiny unit test for the parser
