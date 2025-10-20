import json,os,sys
sys.path.append(os.path.dirname(__file__))
from excel_scan import pass_into_kb as excel_process
from pdf_scan.run_pdf_scan import resolve_alert_contacts
import Knowledge_Base   
    
def main(raw='test'): #raw text parameter 
    print(raw, end="\n\n RAW RAW --------------------------------------------------------")
    tuple_result = excel_process.main(raw)#txt input parameter
    json_path = Knowledge_Base.main(tuple_result)
    print(resolve_alert_contacts())
    #pdf(escalation)
    #output to json back to main main
    return json_path , resolve_alert_contacts()
if __name__ == "__main__":
    main()