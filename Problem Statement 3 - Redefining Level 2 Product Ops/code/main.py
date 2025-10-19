from email_processor.email_processor import process_any
from processing_folder import processing_main
from db_core_test import run_connector
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
    print(result)

    #processing main , raw text as input
    json_path , pdf_output = processing_main.main(raw)
    #complie output from docx(json) , pdf() , db 
    #print(json_path)
    #print(pdf_output)
    #llm 
    
    #output 
    
    return None


if __name__ == "__main__":
    
    main()