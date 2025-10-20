from email_processor.email_processor import process_any
from processing_folder import processing_main
from db_core_test import run_connector
from compiler.result_pdf_compiler import compile_results
from LLM_folder.llm_decision_maker import decide_solution
import json
import sys

# Redirect all print to stderr so only JSON goes to stdout
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def run(text: str) -> dict:
    """
    Main pipeline that processes incident text through the entire workflow.
    Returns a dictionary with all results.
    """
    try:
        # 1. Email processing - extract structured data from raw text
        problem_draft = process_any(text) 

        # 2. Database query - find matching records
        db_result = run_connector(problem_draft)
        
        # 3. Processing main - Knowledge Base search and PDF contact extraction
        json_path, pdf_output = processing_main.main(problem_draft)
        
        # 4. Compile all results together
        compiled, compiled_path = compile_results(db_result, json_path, pdf_output)
        
        # 5. LLM decision maker - final recommendation
        try:
            decision = decide_solution(compiled_path, raw_email=text)
        except Exception as e:
            decision = {"error": f"decide_solution failed: {str(e)}"}
        
        # 6. Read the human-readable summary text file if it exists
        human_readable_text = None
        try:
            from pathlib import Path
            if compiled_path:
                compiled_file_path = Path(compiled_path)
                human_file_path = compiled_file_path.with_name("final_solution_human.txt")
                if human_file_path.exists():
                    human_readable_text = human_file_path.read_text(encoding='utf-8')
        except Exception as e:
            # If reading fails, just continue without it
            pass
        
        # Return complete results
        return {
            "problem_draft": problem_draft,
            "db_result": db_result,
            "json_path": json_path,
            "pdf_output": pdf_output,
            "compiled": compiled,
            "compiled_path": compiled_path,
            "decision": decision,
            "human_readable": human_readable_text
        }
    except Exception as e:
        return {
            "error": f"Pipeline failed: {str(e)}",
            "stage": "unknown"
        }

if __name__ == "__main__":
    # Redirect all stdout to stderr so only final JSON goes to stdout
    original_stdout = sys.stdout
    sys.stdout = sys.stderr
    
    try:
        # Check if input is provided via stdin (from Node.js proxy)
        if not sys.stdin.isatty():
            # Read from stdin
            raw_email = sys.stdin.read().strip()
        else:
            # Default test case for local testing
            raw_email = (
                "Email ALR-861600 | REF-COP-0001 - Duplicate Container information received. "
                "Hi Jen, Please assist in checking container CMAU0000020. "
                "Customer on PORTNET is seeing 2 identical containers information."
            )
        
        # Run the pipeline
        result = run(raw_email)
        
        # Restore stdout and output JSON
        sys.stdout = original_stdout
        print(json.dumps(result, default=str, ensure_ascii=False))
    finally:
        # Ensure stdout is restored even if there's an error
        sys.stdout = original_stdout