from email_processor.email_processor import process_any
from processing_folder import processing_main
from db_core_test import run_connector
from compiler.result_pdf_compiler import compile_results
from LLM_folder.llm_decision_maker import decide_solution
import json

def main():
    #read input 
    raw_email = (
    "Email ALR-861600 | REF-COP-0001 - Duplicate Container information received. "
    "Hi Jen, Please assist in checking container REF-COP-0001. "
    "Customer on PORTNET is seeing 2 identical containers information."
    )
    #email processing , output raw text 
    raw = process_any(raw_email)
    
    #db
    result = run_connector(raw)
    #print(result)

    #processing main , raw text as input
    json_path , pdf_output = processing_main.main(raw)
    
    #complie output from docx(json) , pdf() , db 
    #print(json_path)
    #print(pdf_output)
    compiled, compiled_path = compile_results(result, json_path, pdf_output)


    # optional: print short preview
    #print("\n🧩 Compiled summary:")
    #print(json.dumps(compiled, indent=2), "...\n")

    #llm 
    decision = decide_solution(compiled_path, raw_email=raw_email)

    print("\n🧠 LLM Decision:\n", json.dumps(decision, indent=2))




    #output 
    
    return None


if __name__ == "__main__":
    main()