import json,os,sys
sys.path.append(os.path.dirname(__file__))
import excel_scan.pass_into_kb as excel_process
import Knowledge_Base
def read_matching_count(): 
    with open("result.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Access the count value
    count_value = data.get("count", 0)

    print("Count:", count_value)
    
def main():
    tuple_result = excel_process.main()
    json_path = Knowledge_Base.main(tuple_result)
    #pdf(escalation)
    #output to json back to main main
    return None
if __name__ == "__main__":
    main()