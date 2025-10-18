"""Connector: read an input string, process with email_processor, query seed JSONs, and print results.

Usage:
  # interactive
  python code/main.py
  # or pipe a message
  echo "EDI REF-IFT-0007 stuck..." | python code/main.py
"""
from pathlib import Path
import importlib.util
import json
import sys


def load_module_from_path(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)
	# Ensure the module is importable by name while executing (fixes dataclass/name resolution)
	import sys as _sys
	_sys.modules[name] = mod
	spec.loader.exec_module(mod)
	return mod


def main():
	base = Path(__file__).resolve().parent  # code/
	email_path = base / 'email_processor' / 'email_processor.py'
	query_path = base / 'db_folder' / 'query_seed.py'

	if not email_path.exists():
		print(f'email_processor not found at {email_path}')
		return
	if not query_path.exists():
		print(f'query_seed not found at {query_path}')
		return

	email_mod = load_module_from_path('email_processor', email_path)
	query_mod = load_module_from_path('query_seed', query_path)

	# Read input string: if piped, read all stdin; otherwise interactive prompt
	if not sys.stdin.isatty():
		text = sys.stdin.read().strip()
	else:
		try:
			text = input('Enter email text (single line) or paste and press Enter: ').strip()
		except EOFError:
			text = ''

	if not text:
		print(json.dumps({"error": "No input provided"}, indent=2))
		sys.exit(1)

	# Process email to extract ProblemDraft
	try:
		pd = email_mod.process_any(text)
	except Exception as e:
		print(json.dumps({"error": f"Email processing failed: {e}"}, indent=2))
		sys.exit(1)

	# pd is a ProblemDraft dataclass instance; convert to dict if possible
	try:
		pd_dict = pd.to_dict()
	except Exception:
		# fallback: try as dict
		pd_dict = pd if isinstance(pd, dict) else {}

	print('\n-- Extracted ProblemDraft --')
	print(json.dumps(pd_dict, indent=2, ensure_ascii=False))

	# Prepare datasets from query module
	vessels = query_mod.load_json(query_mod.FILES['vessels'])
	containers = query_mod.load_json(query_mod.FILES['containers'])
	edi = query_mod.load_json(query_mod.FILES['edi_messages'])
	api = query_mod.load_json(query_mod.FILES['api_events'])

	results = []

	vars = pd_dict.get('variables', {})
	# extract lists (may be empty)
	cntrs = vars.get('cntr_no') or []
	msgs = vars.get('message_ref') or []
	vnames = vars.get('vessel_name') or []
	corrs = vars.get('correlation_id') or []

	# Query each and collect matches
	for c in cntrs:
		results += query_mod.find_by_key(containers, 'cntr_no', c)
	for m in msgs:
		results += query_mod.find_by_key(edi, 'message_ref', m)
	for v in vnames:
		results += query_mod.find_by_key(vessels, 'vessel_name', v)
	for r in corrs:
		results += query_mod.find_by_key(api, 'correlation_id', r)

	# Deduplicate by primary identifiers to avoid duplicate container snapshots
	def _id_key(obj: dict) -> str:
		# Prefer strong identifiers where available
		if 'cntr_no' in obj and obj.get('cntr_no'):
			return f"cntr:{obj.get('cntr_no')}"
		if 'message_ref' in obj and obj.get('message_ref'):
			return f"msg:{obj.get('message_ref')}"
		if 'correlation_id' in obj and obj.get('correlation_id'):
			return f"corr:{obj.get('correlation_id')}"
		if 'imo_no' in obj and obj.get('imo_no'):
			return f"imo:{obj.get('imo_no')}"
		if 'vessel_name' in obj and obj.get('vessel_name'):
			return f"vsl:{obj.get('vessel_name')}"
		# Fallback to JSON fingerprint
		return f"json:{json.dumps(obj, sort_keys=True, ensure_ascii=False)}"

	seen = set()
	unique = []
	for r in results:
		k = _id_key(r)
		if k not in seen:
			seen.add(k)
			unique.append(r)

	print('\n-- Query results --')
	if not unique:
		print(json.dumps({"error": "No matches found for extracted variables"}, indent=2))
		sys.exit(1)

	print(json.dumps(unique, indent=2, ensure_ascii=False))


if __name__ == '__main__':
	main()