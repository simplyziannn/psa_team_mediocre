PSA Mediocre — Operations Automation
Overview

This project automates the incident triage and resolution process for PSA Operations using a combination of LLM-based natural language understanding and document matching across multiple data sources.

Given an email or alert (test case), the system automatically scans Excel logs, DOCX SOP files, and PDF escalation contacts—along with referencing the database—to produce relevant problem statements, solutions, and SOPs.

System Flow
1. Input

A test case or email is provided as raw input.

The LLM processes and structures the information into a standardized problem format.

2. Processing Main

The core processing module performs multi-source scanning:

Excel Processing → Searches entire Excel logs for related incidents.

Docx Processing → Extracts SOPs and resolution procedures.

PDF Processing → Reads escalation contacts and team directories.

The results are passed as structured data:
problem_statement + case_log + contacts

3. Compiler

The compiler consolidates all processed outputs and sends a structured summary back to the LLM for reasoning.

4. LLM Decision Engine

The LLM interprets the compiled data to:

Generate solutions and/or SOPs based on the case log and problem statement.

Cross-reference escalation contacts from PDF sources.

Produce relevant escalation steps, resolutions, and standard procedures.

Problem Statement 3 - Redefining Level 2 Product Ops/
│
├── code/
│   ├── main.py                     # Entry point
│   ├── db_core.py                  # DB utilities
│   ├── db_core_test.py             # Testing DB modules
│   │
│   ├── compiler/
│   │   └── result_pdf_compiler.py  # Compiles final structured output
│   │
│   ├── db_folder/                  # Static seed JSONs + query helpers
│   │   ├── extract_seed_json.py
│   │   ├── query_seed.py
│   │   ├── output/
│   │   │   ├── api_events.json
│   │   │   ├── containers.json
│   │   │   ├── edi_messages.json
│   │   │   └── vessels.json
│   │   └── README.md
│   │
│   ├── email_processor/
│   │   └── email_processor.py      # Parses and normalizes email input
│   │
│   ├── LLM_folder/
│   │   ├── call_openai_basic.py    # LLM API helper
│   │   ├── embedding_helper.py     # Embedding + cosine similarity utils
│   │   ├── llm_decision_maker.py   # Decision engine for solutions/SOPs
│   │   └── prompt_config.json
│   │
│   ├── processing_folder/
│   │   ├── processing_main.py      # Central orchestrator
│   │   ├── Knowledge_Base.py
│   │   │
│   │   ├── excel_scan/
│   │   │   ├── excel_scanner.py
│   │   │   ├── excel_llm_reader.py
│   │   │   ├── pass_into_kb.py
│   │   │   ├── extract_text_sample.py
│   │   │   ├── caching.py
│   │   │   ├── embedding_helper.py
│   │   │   ├── Case Log.xlsx
│   │   │   └── case_log.embcache.pkl
│   │   │
│   │   ├── doc_scan/
│   │   │   ├── docx_scanner.py
│   │   │   └── DOCX DEBUG JSON OUTPUT/
│   │   │       └── debugging_alert_result.json
│   │   │
│   │   ├── pdf_scan/
│   │   │   ├── run_pdf_scan.py
│   │   │   ├── contacts_lookup.py
│   │   │   ├── json_extract.py
│   │   │   ├── contacts.json
│   │   │   └── Product Team Escalation Contacts.pdf
│   │   │
│   │   └── JSON OUTPUT/
│   │       ├── alert_result.json
│   │       ├── compiled_output.json
│   │       ├── final_solution.json
│   │       └── final_solution_human.txt
│   │
│   └── __init__.py
│
├── Info/
│   ├── Case Log.xlsx
│   ├── Knowledge Base.docx
│   ├── Product Team Escalation Contacts.pdf
│   ├── Test Cases.pdf
│   ├── Application Logs/
│   │   ├── api_event_service.log
│   │   ├── vessel_registry_service.log
│   │   └── ...
│   └── Database/
│       ├── db.sql
│       └── SCHEMA_OVERVIEW.md
│
└── Code Sprint 2025 Problem Statements.pdf


Example Workflow

Input → "EDI REF-COP-0001 stuck without ACK"

Email Processor normalizes and structures the text.

Processing Main executes:

Excel: Finds matching case logs.

Docx: Retrieves related SOP steps.

PDF: Identifies escalation contacts.

Compiler merges results into a unified JSON.

LLM outputs:

Root cause

Resolution

SOP reference

Escalation path

Tech Stack

Language: Python 3.13

Core Libraries: openpyxl, python-docx, PyPDF2, pandas

LLM Integration: OpenAI GPT API

Similarity Search: Embeddings + cosine similarity

Output Format: JSON and human-readable text

Output Examples
File	Description
alert_result.json	LLM-parsed incident
compiled_output.json	Merged data across all sources
final_solution.json	Structured result (root cause, solution, SOP)
final_solution_human.txt	Summarized human-readable output
Team Mediocre

PSA Hackathon 2025 — Problem Statement 3: Redefining Level 2 Product Ops
Leveraging AI to streamline incident management, automate triage, and accelerate root cause identification.
