# PSA Level 2 Product Ops - AI Coding Instructions

## Project Overview
This is a **port transhipment system** for PSA managing vessel arrivals, container movements, and EDI message integration. The core domain models Singapore's container terminal operations with vessel scheduling, container lifecycle tracking, and partner communication.

## Architecture Patterns

### Database-Centric Design
- **MySQL 8.0** schema in `Info/Database/db.sql` with comprehensive seed data
- **Versioned containers**: Uses composite PK `(cntr_no, created_at)` for historical snapshots
- **Active-only constraints**: `vessel_advice` uses generated columns to enforce single active advice per system name
- **Status enums**: Container status flow: `GATE_IN` → `IN_YARD` → `LOADED`/`DISCHARGED` → `TRANSHIP` → `GATE_OUT`

### Service Architecture 
Services follow naming pattern `{domain}_service.log` in `Info/Application Logs/`:
- `container_service`: Manages container snapshots with versioning warnings
- `vessel_advice_service`: Handles arrival/departure advice lifecycle  
- `edi_advice_service`: EDI message processing (COPARN, COARRI, CODECO, IFTMCS, IFTMIN)
- `api_event_service`: External system events with JSON payloads
- `berth_application_service`: Berth planning linked to vessel advice

### LLM Integration
- **Azure API**: Uses `LLM_folder/call_openai_basic.py` with PSA-specific endpoint
- **Environment**: API key in `.env` file (`PORTAL_SUB_KEY`)
- **Error handling**: Returns formatted HTTP errors and network exceptions

## Key Data Patterns

### Container Lifecycle
```sql
-- Always query latest snapshot per container
SELECT * FROM container WHERE cntr_no=? ORDER BY created_at DESC LIMIT 1
```

### UN/LOCODE Usage
- Port codes are 5-char UN/LOCODE format: `SGSIN` (Singapore), `CNSHA` (Shanghai), `HKHKG` (Hong Kong)
- Used in `vessel.last_port/next_port` and `container.origin_port/destination_port`

### EDI Message Flow
- Direction: `IN` (from carriers) vs `OUT` (to carriers)  
- Status progression: `RECEIVED` → `PARSED` → `ACKED`/`ERROR`
- Message types follow shipping standards: COPARN (booking), COARRI (arrival), CODECO (equipment)

### JSON Event Payloads
API events store operational data in `payload_json`:
- Gate events: `{"gate":"B2","truck":"SGL1234Z"}`
- Load/discharge: `{"stow":"11-06-07"}` (bay-row-tier), `{"crane":"QC-05"}`

## Development Conventions

### Database Operations
- Use `container_id` for foreign keys (surrogate key), not composite PK
- Check `effective_end_datetime IS NULL` for active vessel advice
- Leverage views: `vw_tranship_pipeline` for operational overview, `vw_edi_last` for latest EDI status

### Service Integration
- Correlation IDs format: `corr-{service}-{sequence}` (e.g., `corr-api-0005`)
- HTTP status logging with latency tracking
- Event publishing pattern for container updates

### File Structure
- **Empty placeholder folders**: `compiler/`, `db_folder/`, `excel_folder/` are reserved for future components
- **Info/**: Contains reference data, logs, and documentation
- **code/**: Main application code with modular LLM integration

## Testing with Seed Data
- 20 vessels: MV Lion City series (01-10) and MV Merlion series (11-20)
- Container examples: `MSKU*`, `MSCU*`, `OOLU*`, `TEMU*`, `CMAU*` prefixes
- Deliberate duplicate: `CMAU0000020` has two snapshots for versioning tests
- Vessel advice: Active entries for MV Lion City 07/08, MV Merlion 11/15

## External Dependencies
- **Azure API Management**: PSA-specific GPT-5-mini deployment
- **MySQL 8.0+**: Required for JSON columns and generated column features
- **EDI Standards**: EDIFACT message formats for maritime logistics

When working on this codebase, always consider the port operational context and maritime data standards. Container movements, vessel schedules, and EDI communications follow strict industry protocols that must be preserved in any modifications.