PSA Mediocre — Level 2 Product Ops Automation
Overview

This project automates the incident triage and resolution process for PSA Level 2 Product Ops using a combination of LLM-based natural language understanding and document matching across multiple data sources.

Given an email or alert (test case), the system automatically scans Excel logs, DOCX SOP files, and PDF escalation contacts to produce relevant problem statements, solutions, and SOPs.

⚙️ System Flow
1. Input

A test case or email is provided as raw input.

The LLM processes and structures the information into a standardized problem format.

2. Processing Main

The core processing module performs multi-source scanning:

Excel Processing → Searches case logs for related issues.

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
