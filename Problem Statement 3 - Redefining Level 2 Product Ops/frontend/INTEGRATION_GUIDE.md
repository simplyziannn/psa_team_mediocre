# Frontend ↔ Backend Integration Guide

## 🎯 How It Works

Your frontend is now fully connected to the Python backend pipeline!

### Architecture Flow:

```
Browser (Port 8080)
    ↓
Vite Dev Server (proxy /api → port 5000)
    ↓
Node.js Express Proxy (server.cjs on port 5000)
    ↓
Python subprocess (main.py)
    ↓
Complete Pipeline:
  1. Email Processor (extract incident data)
  2. Database Query (find matching records)
  3. Knowledge Base Search (find relevant SOPs)
  4. PDF Contact Extraction (get escalation contacts)
  5. Compiler (combine all results)
  6. LLM Decision Maker (final recommendation)
    ↓
JSON Output to stdout
    ↓
Node.js captures and returns to browser
    ↓
Frontend displays beautifully formatted results
```

## 🚀 How to Run

### 1. Start the Development Server

```bash
cd frontend
npm run dev
```

This single command starts **both**:
- Node.js proxy server on port 5000
- Vite dev server on port 8080

### 2. Open Browser

Navigate to: `http://localhost:8080`

### 3. Submit an Incident

Try this test case:
```
Please assist in checking container CMAU0000020. 
Customer on PORTNET is seeing 2 identical containers information.
```

## 📊 What You'll See

The frontend will display:

### ✅ Incident Analysis Section:
- **📋 Incident Type**: Automatically detected (e.g., ContainerData)
- **Confidence**: Extraction confidence percentage
- **🎯 Module**: Which PSA module handles this (EDI/API, Container, Vessel, etc.)

### ✅ Resolution Section:
- **📝 Summary**: Brief description of the issue
- **🔍 Root Cause**: What caused the problem
- **🛠️ Resolution Steps**: Numbered list of actions to take
- **📦 Database Matches**: Container details from your MySQL database
  - Container number, type, status
  - Origin → Destination route
  - Vessel information, ETA

### ✅ Escalation Section (if needed):
- **🎯 Target**: Which team to escalate to
- **👥 Contacts**: Names and emails
- **📝 Escalation Steps**: Specific escalation procedures

## 🔧 Pipeline Stages

Your `main.py` now runs this complete workflow:

```python
def run(text: str) -> dict:
    # 1. Email processing - extract structured data
    problem_draft = process_any(text)
    
    # 2. Database query - find matching records
    db_result = run_connector(problem_draft)
    
    # 3. Knowledge Base + PDF contacts
    json_path, pdf_output = processing_main.main(problem_draft)
    
    # 4. Compile all results
    compiled, compiled_path = compile_results(db_result, json_path, pdf_output)
    
    # 5. LLM decision maker
    decision = decide_solution(compiled_path, raw_email=text)
    
    return {
        "problem_draft": problem_draft,
        "db_result": db_result,
        "json_path": json_path,
        "pdf_output": pdf_output,
        "compiled": compiled,
        "compiled_path": compiled_path,
        "decision": decision
    }
```

## 🎨 Frontend Enhancements

The `Index.tsx` now:
- ✅ Parses all pipeline stages
- ✅ Formats output with icons and clear sections
- ✅ Shows database matches with container details
- ✅ Displays escalation contacts and steps
- ✅ Handles errors gracefully
- ✅ Shows confidence levels
- ✅ Dynamic severity badges

## 🐛 Troubleshooting

### If you see errors:

1. **"Python script exited with code X"**
   - Check terminal output for Python errors
   - Ensure all Python dependencies are installed
   - Check that `.env` file exists in `code/LLM_folder/`

2. **"Failed to parse JSON"**
   - Check if Python script printed anything to stdout
   - Look at server.cjs logs in terminal

3. **Connection refused**
   - Make sure `npm run dev` is running
   - Check that ports 5000 and 8080 are not in use

4. **Unicode/Emoji errors**
   - These should be fixed, but if they appear, check Python print statements
   - All emoji should be replaced with ASCII equivalents

## 📁 Key Files

- `frontend/server.cjs` - Node.js proxy that spawns Python
- `frontend/src/pages/Index.tsx` - Main UI with result formatting
- `code/main.py` - Python pipeline orchestrator
- `frontend/vite.config.ts` - Vite proxy configuration

## 🎉 Testing

Test with different incident types:
- Container issues
- EDI message problems
- Vessel scheduling
- API errors

The system will automatically:
- Extract relevant data
- Query the database
- Search knowledge base
- Find escalation contacts
- Provide resolution steps

Enjoy your fully integrated incident processing system! 🚀
