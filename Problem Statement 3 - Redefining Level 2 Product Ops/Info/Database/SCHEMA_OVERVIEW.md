# Schema Documentation

## 1. Purpose & Scope
This document explains the MySQL 8.0 schema contained in `db.sql`.

## 2. High-Level Domain Overview
The schema models a simplified transhipment / terminal integration domain:
- **Vessels** calling the port (static descriptive data)
- **Containers** moving through the terminal with evolving statuses
- **EDI Messages** exchanged with carriers / partners providing event & document flows
- **API Events** internal/external system events (e.g., gate moves, load/discharge) with JSON payloads
- **Vessel Advice** (arrival / port program advice lifecycle) including controlled re‑use of a local vessel name
- **Berth Applications** referencing the active vessel advice (port program / berth planning linkage)
- **Views** for quick operational insights (latest EDI message type & timestamp, container pipeline snapshot)

## 3. Entity-by-Entity Detail
### 3.1 `vessel`
CREATE TABLE vessel (
  vessel_id        BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  imo_no           INT UNSIGNED NOT NULL,
  vessel_name      VARCHAR(100) NOT NULL,
  call_sign        VARCHAR(20),
  operator_name    VARCHAR(100),
  flag_state       VARCHAR(50),
  built_year       SMALLINT,
  capacity_teu     INT,
  loa_m            DECIMAL(6,2),       
  beam_m           DECIMAL(5,2),
  draft_m          DECIMAL(4,2),
  last_port        CHAR(5),            
  next_port        CHAR(5),            
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_vessel_imo (imo_no),
  KEY idx_vessel_next_port (next_port)
) ENGINE=InnoDB;

Stores static / quasi-static vessel reference data.
- `imo_no` unique (international identity).
- Supplemental attributes (dimensions, capacity) support enrichment & analytics.
- Indexed `next_port` to facilitate filtering / scheduling queries.
- `last_port` / `next_port` use UN/LOCODE 5-char codes (e.g., `SGSIN`).
- `next_port` retained for planning even though earlier design considered deprecation.

### 3.2 `container`
CREATE TABLE container (
  container_id     BIGINT UNSIGNED AUTO_INCREMENT,
  cntr_no          VARCHAR(11) NOT NULL,      
  iso_code         CHAR(4) NOT NULL,          
  size_type        VARCHAR(10) NOT NULL,      
  gross_weight_kg  DECIMAL(10,2),
  status           ENUM('IN_YARD','ON_VESSEL','GATE_OUT','GATE_IN','DISCHARGED','LOADED','TRANSHIP') NOT NULL,
  origin_port      CHAR(5) NOT NULL,
  tranship_port    CHAR(5) NOT NULL DEFAULT 'SGSIN',
  destination_port CHAR(5) NOT NULL,
  hazard_class     VARCHAR(10) NULL,          
  vessel_id        BIGINT UNSIGNED,
  eta_ts           DATETIME NULL,
  etd_ts           DATETIME NULL,
  last_free_day    DATE NULL,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (cntr_no, created_at),          
  UNIQUE KEY uk_container_id (container_id),  
  KEY idx_container_vessel (vessel_id),
  KEY idx_container_status (status),
  CONSTRAINT fk_container_vessel FOREIGN KEY (vessel_id) REFERENCES vessel(vessel_id)
    ON UPDATE CASCADE ON DELETE SET NULL
) ENGINE=InnoDB;

Represents container instances with **versioned snapshots** via a composite primary key.
- **Composite PK:** `(cntr_no, created_at)` keeps historical revisions (e.g., status changes captured by re-inserts; current script seeds 20 base containers plus 1 deliberate later version row for `CMAU0000020`).
- `iso_code` examples: `22G1`, `45R1`; `size_type` examples: `20GP`, `40HQ`, `45RF`.
- `hazard_class` holds IMDG / UN class codes (e.g., `3`, `8`, `9`).
- Retains `container_id` surrogate unique key for external FKs (no need to know timestamp to reference latest snapshot).
- Status ENUM values model operational lifecycle: yard presence, gate moves, vessel ops, tranship.

### 3.3 `edi_message`
CREATE TABLE edi_message (
  edi_id           BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  container_id     BIGINT UNSIGNED, 
  vessel_id        BIGINT UNSIGNED, 
  message_type     ENUM('COPARN','COARRI','CODECO','IFTMCS','IFTMIN') NOT NULL,
  direction        ENUM('IN','OUT') NOT NULL,
  status           ENUM('RECEIVED','PARSED','ACKED','ERROR') NOT NULL DEFAULT 'RECEIVED',
  message_ref      VARCHAR(50) NOT NULL,
  sender           VARCHAR(100) NOT NULL,
  receiver         VARCHAR(100) NOT NULL,
  sent_at          DATETIME NOT NULL,
  ack_at           DATETIME NULL,
  error_text       VARCHAR(500) NULL,
  raw_text         MEDIUMTEXT NULL,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_edi_container (container_id),
  KEY idx_edi_vessel (vessel_id),
  KEY idx_edi_type_time (message_type, sent_at),
  CONSTRAINT fk_edi_container FOREIGN KEY (container_id) REFERENCES container(container_id)
    ON UPDATE CASCADE ON DELETE SET NULL,
  CONSTRAINT fk_edi_vessel FOREIGN KEY (vessel_id) REFERENCES vessel(vessel_id)
    ON UPDATE CASCADE ON DELETE SET NULL
) ENGINE=InnoDB;

Captures inbound / outbound EDI messages (COPARN, COARRI, CODECO, IFTMCS, IFTMIN).
- Typical flow: booking (COPARN) → arrival/departure (COARRI) → gate/yard (CODECO) → transport instructions (IFT*).
- Lifecycle state machine: `RECEIVED` → `PARSED` → `ACKED` / `ERROR`.
- `raw_text` retains original interchange; `error_text` stores concise parse/validation issues.

### 3.4 `api_event`
CREATE TABLE api_event (
  api_id           BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  container_id     BIGINT UNSIGNED,
  vessel_id        BIGINT UNSIGNED,
  event_type       ENUM('GATE_IN','GATE_OUT','LOAD','DISCHARGE','CUSTOMS_CLEAR','HOLD','RELEASE') NOT NULL,
  source_system    VARCHAR(50) NOT NULL,    
  http_status      SMALLINT,
  correlation_id   VARCHAR(64),
  event_ts         DATETIME NOT NULL,
  payload_json     JSON,
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_api_container (container_id),
  KEY idx_api_event_type_time (event_type, event_ts),
  CONSTRAINT fk_api_container FOREIGN KEY (container_id) REFERENCES container(container_id)
    ON UPDATE CASCADE ON DELETE SET NULL,
  CONSTRAINT fk_api_vessel FOREIGN KEY (vessel_id) REFERENCES vessel(vessel_id)
    ON UPDATE CASCADE ON DELETE SET NULL
) ENGINE=InnoDB;

Represents API-sourced operational events (e.g., gate or equipment triggers).
- `source_system` examples: `DG-BOT`, `TOS`, `CMS`.
- `payload_json` commonly holds: bay/row/tier for stowage, truck plate for gate, crane ID for moves.
- Correlation IDs enable distributed tracing across microservices/log pipelines.

### 3.5 `vessel_advice`
CREATE TABLE vessel_advice (
  vessel_advice_no        BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  vessel_name          VARCHAR(100) NOT NULL,
  system_vessel_name          VARCHAR(20) NOT NULL,          
  effective_start_datetime         DATETIME NOT NULL,
  effective_end_datetime           DATETIME NULL,                 
  created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

  system_vessel_name_active   VARCHAR(20) AS (CASE WHEN effective_end_datetime IS NULL THEN system_vessel_name ELSE NULL END) STORED,
  UNIQUE KEY uk_system_vessel_name_active (system_vessel_name_active),  
  KEY idx_vessel_advice_name_hist (system_vessel_name, effective_start_datetime) 
) ENGINE=InnoDB;

Models arrival / port program “advice” with controlled reuse of `system_vessel_name`.
- At most **one ACTIVE** (non‑expired) advice per local name.
- Generated stored column technique: `system_vessel_name_active = CASE WHEN effective_end_datetime IS NULL THEN system_vessel_name ELSE NULL END`.
- Unique index on generated column enforces active-only uniqueness while allowing unlimited historical rows (NULL bypasses uniqueness).

### 3.6 `berth_application`
CREATE TABLE berth_application (
  application_no          BIGINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
  vessel_advice_no        BIGINT UNSIGNED NOT NULL,
  vessel_close_datetime   DATETIME NULL,
  deleted                 CHAR(1) NOT NULL DEFAULT 'N',
  berthing_status         CHAR(1) NOT NULL DEFAULT 'A',
  created_at              TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_ba_vsl_advice (vessel_advice_no),
  CONSTRAINT fk_berth_application_vessel_advice FOREIGN KEY (vessel_advice_no) REFERENCES vessel_advice(vessel_advice_no)
    ON UPDATE CASCADE ON DELETE CASCADE
) ENGINE=InnoDB;

Links to the active advice (berth / program application) and inherits lifecycle constraints.
- `deleted` soft flag: current values `N` (Not deleted / active), `A` (Archived – logically retired but retained for history). If a hard delete is required, consider archiving first, then purging in maintenance windows.
- `berthing_status` initial code set: `A` (Active). Future candidates could include `C` (Closed), `X` (Cancelled), `H` (On Hold) depending on workflow expansion.
- Foreign key cascade ensures removal when advice is hard-deleted (operationally prefer expiring advice over deletion).

### 3.7 Views
#### 3.7.1 `vw_tranship_pipeline`
CREATE VIEW vw_tranship_pipeline AS
SELECT
  c.cntr_no,
  c.size_type,
  c.status,
  c.origin_port,
  c.tranship_port,
  c.destination_port,
  v.vessel_name,
  v.imo_no,
  c.eta_ts,
  c.etd_ts,
  c.last_free_day
FROM container c
LEFT JOIN vessel v ON v.vessel_id = c.vessel_id;

Operational snapshot combining container routing/status with vessel identity.
#### 3.7.2 `vw_edi_last`
CREATE VIEW vw_edi_last AS
SELECT
  c.cntr_no,
  MAX(e.sent_at) AS last_edi_time,
  SUBSTRING_INDEX(GROUP_CONCAT(e.message_type ORDER BY e.sent_at DESC), ',', 1) AS last_edi_type,
  SUBSTRING_INDEX(GROUP_CONCAT(e.status ORDER BY e.sent_at DESC), ',', 1) AS last_edi_status
FROM edi_message e
JOIN container c ON c.container_id = e.container_id
GROUP BY c.cntr_no;

Rollup to latest EDI message type and status per container via ordered GROUP_CONCAT trick.
