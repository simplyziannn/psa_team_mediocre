-- Schema: transhipment domain (MySQL 8.0+)
CREATE DATABASE IF NOT EXISTS appdb CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE appdb;


SET FOREIGN_KEY_CHECKS = 0;

drop view if exists vw_edi_last;

drop view if exists vw_tranship_pipeline;

drop table if exists api_event;      

drop table if exists edi_message;    

drop table if exists container;     

drop table if exists vessel;         

SET FOREIGN_KEY_CHECKS = 1;
-- VESSEL
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

-- Seed 20 vessels -- some other lines
INSERT INTO vessel (imo_no, vessel_name, call_sign, operator_name, flag_state, built_year, capacity_teu, loa_m, beam_m, draft_m, last_port, next_port)
VALUES
(9300001,'MV Lion City 01','9VLC1','Oceanic Shipping','Singapore',2010,14000,366.00,51.00,15.00,'CNSHA','SGSIN'),
(9300002,'MV Lion City 02','9VLC2','BlueWave Lines','Panama',2011,14500,368.50,51.20,15.20,'HKHKG','CNSHA'),
(9300003,'MV Lion City 03','9VLC3','HarborStar','Liberia',2012,15000,370.00,52.00,15.50,'SGSIN','MYTPP'),
(9300004,'MV Lion City 04','9VLC4','Oceanic Shipping','Marshall Islands',2013,15500,372.00,52.10,15.60,'JPTYO','SGSIN'),
(9300005,'MV Lion City 05','9VLC5','BlueWave Lines','Denmark',2014,16000,375.00,53.00,15.80,'SGSIN','HKHKG'),
(9300006,'MV Lion City 06','9VLC6','HarborStar','Malta',2015,16500,377.00,53.50,16.00,'CNSZX','CNSHA'),
(9300007,'MV Lion City 07','9VLC7','Trident Global','Hong Kong',2016,17000,380.00,54.00,16.00,'SGSIN','IDJKT'),
(9300008,'MV Lion City 08','9VLC8','Oceanic Shipping','Singapore',2017,17500,382.00,54.20,16.20,'MYTPP','SGSIN'),
(9300009,'MV Lion City 09','9VLC9','BlueWave Lines','Germany',2018,18000,384.00,55.00,16.50,'SGSIN','JPTYO'),
(9300010,'MV Lion City 10','9VLA0','HarborStar','UK',2019,18500,386.00,55.10,16.50,'KRPTK','SGSIN'),
(9300011,'MV Merlion 11','9VML1','Trident Global','Singapore',2020,19000,388.00,56.00,16.50,'SGSIN','HKHKG'),
(9300012,'MV Merlion 12','9VML2','Oceanic Shipping','Panama',2020,19000,388.00,56.00,16.50,'HKHKG','SGSIN'),
(9300013,'MV Merlion 13','9VML3','BlueWave Lines','Liberia',2021,19500,390.00,56.50,16.60,'CNSHA','SGSIN'),
(9300014,'MV Merlion 14','9VML4','HarborStar','Marshall Islands',2021,19500,390.00,56.50,16.60,'SGSIN','CAXMN'),
(9300015,'MV Merlion 15','9VML5','Trident Global','Denmark',2022,20000,395.00,57.00,16.80,'SGSIN','USLAX'),
(9300016,'MV Merlion 16','9VML6','Oceanic Shipping','Malta',2022,20000,395.00,57.00,16.80,'USLAX','SGSIN'),
(9300017,'MV Merlion 17','9VML7','BlueWave Lines','Hong Kong',2023,20500,398.00,58.00,17.00,'SGSIN','AEMAA'),
(9300018,'MV Merlion 18','9VML8','HarborStar','Singapore',2023,20500,398.00,58.00,17.00,'AEMAA','SGSIN'),
(9300019,'MV Merlion 19','9VML9','Trident Global','Germany',2024,21000,400.00,58.60,17.20,'SGSIN','INNSA'),
(9300020,'MV Merlion 20','9VMA0','Oceanic Shipping','Liberia',2024,21000,400.00,58.60,17.20,'INNSA','SGSIN');

-- CONTAINER
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

-- Seed 20 containers 
INSERT INTO container
(cntr_no, iso_code, size_type, gross_weight_kg, status, origin_port, tranship_port, destination_port, hazard_class, vessel_id, eta_ts, etd_ts, last_free_day, created_at)
VALUES
('MSKU0000001','22G1','20GP', 12000,'TRANSHIP','CNSHA','SGSIN','MYTPP',NULL, 1,'2025-10-04 12:00','2025-10-05 18:00','2025-10-10', NOW()),
('MSKU0000002','45R1','40RF',  8000,'IN_YARD','HKHKG','SGSIN','IDJKT',NULL, 2,'2025-10-05 08:00','2025-10-06 20:00','2025-10-11', NOW()),
('MSKU0000003','22G1','20GP', 11000,'DISCHARGED','CNSZX','SGSIN','SGSIN',NULL, 3,'2025-10-03 16:00',NULL,'2025-10-09', NOW()),
('MSKU0000004','22G1','20GP', 13000,'LOADED','MYTPP','SGSIN','JPTYO',NULL, 4,'2025-10-06 10:00','2025-10-06 22:00','2025-10-12', NOW()),
('MSKU0000005','45G1','40HQ', 15000,'TRANSHIP','JPTYO','SGSIN','CNSHA',NULL, 5,'2025-10-04 18:00','2025-10-05 23:00','2025-10-10', NOW()),
('MSCU0000006','45G1','40HQ', 14500,'IN_YARD','SGSIN','SGSIN','USLAX',NULL, 6,'2025-10-07 06:00','2025-10-08 02:00','2025-10-13', NOW()),
('MSCU0000007','22G1','20GP', 10000,'TRANSHIP','CNSHA','SGSIN','HKHKG',NULL, 7,'2025-10-05 05:00','2025-10-05 20:00','2025-10-11', NOW()),
('MSCU0000008','22G1','20GP', 11500,'GATE_IN','HKHKG','SGSIN','SGSIN',NULL, 8,'2025-10-04 09:00',NULL,'2025-10-09', NOW()),
('MSCU0000009','22G1','20GP', 12500,'ON_VESSEL','SGSIN','SGSIN','CAXMN',NULL, 9,NULL,'2025-10-06 12:00','2025-10-12', NOW()),
('MSCU0000010','45R1','40RF',  9000,'TRANSHIP','CAXMN','SGSIN','SGSIN','9', 10,'2025-10-05 14:00','2025-10-06 14:00','2025-10-11', NOW()),
('OOLU0000011','45G1','40HQ', 14800,'TRANSHIP','KRPTK','SGSIN','AEMAA',NULL, 1,'2025-10-06 02:00','2025-10-07 01:00','2025-10-13', NOW()),
('OOLU0000012','22G1','20GP', 10800,'DISCHARGED','AEMAA','SGSIN','SGSIN',NULL, 2,'2025-10-03 22:00',NULL,'2025-10-09', NOW()),
('OOLU0000013','22G1','20GP', 11800,'LOADED','SGSIN','SGSIN','INNSA',NULL, 3,NULL,'2025-10-05 21:00','2025-10-12', NOW()),
('OOLU0000014','22G1','20GP', 11200,'GATE_OUT','SGSIN','SGSIN','SGSIN',NULL, 4,NULL,NULL,'2025-10-08', NOW()),
('TEMU0000015','45G1','40HQ', 15100,'TRANSHIP','INNSA','SGSIN','CNSHA',NULL, 5,'2025-10-04 20:00','2025-10-06 00:00','2025-10-11', NOW()),
('TEMU0000016','45R1','40RF',  8600,'IN_YARD','CNSHA','SGSIN','MYTPP','3', 6,'2025-10-07 08:00','2025-10-08 03:00','2025-10-13', NOW()),
('TEMU0000017','22G1','20GP', 12200,'TRANSHIP','HKHKG','SGSIN','JPTYO',NULL, 7,'2025-10-05 11:00','2025-10-05 23:59','2025-10-11', NOW()),
('TEMU0000018','22G1','20GP', 11900,'ON_VESSEL','JPTYO','SGSIN','SGSIN',NULL, 8,NULL,'2025-10-06 03:00','2025-10-12', NOW()),
('CMAU0000019','45G1','40HQ', 14950,'TRANSHIP','USLAX','SGSIN','SGSIN',NULL, 9,'2025-10-05 16:00','2025-10-06 18:00','2025-10-11', NOW()),
('CMAU0000020','22G1','20GP', 10990,'DISCHARGED','CAXMN','SGSIN','SGSIN',NULL, 10,'2025-10-03 13:00',NULL,'2025-10-09', NOW()),
('CMAU0000020','22G1','20GP', 10990,'DISCHARGED','CAXMN','SGSIN','SGSIN',NULL, 10,'2025-10-03 13:00',NULL,'2025-10-09', NOW() + INTERVAL 1 SECOND);


-- EDI_MESSAGE
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

-- Seed 20 EDI messages
INSERT INTO edi_message
(container_id, vessel_id, message_type, direction, status, message_ref, sender, receiver, sent_at, ack_at, error_text, raw_text)
VALUES
(1,1,'COPARN','IN','PARSED','REF-COP-0001','LINE-PSA','PSA-TOS','2025-10-03 08:01','2025-10-03 08:02',NULL,'UNA:+.? \nUNB+...'),
(2,2,'COPARN','IN','PARSED','REF-COP-0002','LINE-PSA','PSA-TOS','2025-10-03 08:05','2025-10-03 08:06',NULL,'UNA:+.? \nUNB+...'),
(3,3,'COARRI','OUT','ACKED','REF-ARR-0003','PSA-TOS','LINE-PSA','2025-10-03 17:10','2025-10-03 17:12',NULL,'UNH+...'),
(4,4,'COARRI','OUT','ACKED','REF-ARR-0004','PSA-TOS','LINE-PSA','2025-10-04 06:40','2025-10-04 06:41',NULL,'UNH+...'),
(5,5,'CODECO','OUT','ACKED','REF-DEC-0005','PSA-DEPOT','LINE-PSA','2025-10-04 09:00','2025-10-04 09:01',NULL,'UNH+...'),
(6,6,'IFTMIN','IN','PARSED','REF-IFT-0006','LINE-PSA','PSA-TOS','2025-10-04 12:20','2025-10-04 12:21',NULL,'UNH+...'),
(7,7,'IFTMIN','IN','ERROR','REF-IFT-0007','LINE-PSA','PSA-TOS','2025-10-04 12:25',NULL,'Segment missing','UNH+...'),
(8,8,'COPARN','IN','PARSED','REF-COP-0008','LINE-PSA','PSA-TOS','2025-10-04 13:10','2025-10-04 13:11',NULL,'UNH+...'),
(9,9,'COARRI','OUT','ACKED','REF-ARR-0009','PSA-TOS','LINE-PSA','2025-10-04 14:33','2025-10-04 14:34',NULL,'UNH+...'),
(10,10,'CODECO','OUT','ACKED','REF-DEC-0010','PSA-DEPOT','LINE-PSA','2025-10-04 15:00','2025-10-04 15:02',NULL,'UNH+...'),
(11,1,'COPARN','IN','PARSED','REF-COP-0011','LINE-PSA','PSA-TOS','2025-10-05 07:05','2025-10-05 07:06',NULL,'UNH+...'),
(12,2,'IFTMIN','IN','PARSED','REF-IFT-0012','LINE-PSA','PSA-TOS','2025-10-05 07:15','2025-10-05 07:16',NULL,'UNH+...'),
(13,3,'COARRI','OUT','ACKED','REF-ARR-0013','PSA-TOS','LINE-PSA','2025-10-05 08:20','2025-10-05 08:21',NULL,'UNH+...'),
(14,4,'CODECO','OUT','ACKED','REF-DEC-0014','PSA-DEPOT','LINE-PSA','2025-10-05 09:20','2025-10-05 09:22',NULL,'UNH+...'),
(15,5,'IFTMIN','IN','PARSED','REF-IFT-0015','LINE-PSA','PSA-TOS','2025-10-05 10:10','2025-10-05 10:11',NULL,'UNH+...'),
(16,6,'COPARN','IN','PARSED','REF-COP-0016','LINE-PSA','PSA-TOS','2025-10-05 10:30','2025-10-05 10:31',NULL,'UNH+...'),
(17,7,'COARRI','OUT','ACKED','REF-ARR-0017','PSA-TOS','LINE-PSA','2025-10-05 11:00','2025-10-05 11:02',NULL,'UNH+...'),
(18,8,'CODECO','OUT','ACKED','REF-DEC-0018','PSA-DEPOT','LINE-PSA','2025-10-05 12:00','2025-10-05 12:01',NULL,'UNH+...'),
(19,9,'IFTMIN','IN','PARSED','REF-IFT-0019','LINE-PSA','PSA-TOS','2025-10-05 12:30','2025-10-05 12:31',NULL,'UNH+...'),
(20,10,'COPARN','IN','PARSED','REF-COP-0020','LINE-PSA','PSA-TOS','2025-10-05 13:00','2025-10-05 13:02',NULL,'UNH+...');

-- API_EVENT
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

-- Seed 20 API events
INSERT INTO api_event
(container_id, vessel_id, event_type, source_system, http_status, correlation_id, event_ts, payload_json)
VALUES
(1,1,'DISCHARGE','TOS',200,'corr-0001','2025-10-03 17:20', JSON_OBJECT('bay','12','row','04','tier','06')),
(2,2,'GATE_IN','CMS',201,'corr-0002','2025-10-03 18:05', JSON_OBJECT('gate','C3')),
(3,3,'DISCHARGE','TOS',200,'corr-0003','2025-10-03 18:30', JSON_OBJECT('crane','QC-05')),
(4,4,'LOAD','TOS',200,'corr-0004','2025-10-04 06:55', JSON_OBJECT('stow','23-08-04')),
(5,5,'LOAD','TOS',200,'corr-0005','2025-10-04 23:10', JSON_OBJECT('stow','25-02-02')),
(6,6,'GATE_IN','CMS',200,'corr-0006','2025-10-05 01:15', JSON_OBJECT('truck','SGL1234Z')),
(7,7,'LOAD','TOS',200,'corr-0007','2025-10-05 05:25', JSON_OBJECT('stow','11-06-07')),
(8,8,'GATE_OUT','CMS',200,'corr-0008','2025-10-05 08:40', JSON_OBJECT('gate','A1')),
(9,9,'LOAD','TOS',200,'corr-0009','2025-10-05 12:05', JSON_OBJECT('stow','07-10-03')),
(10,10,'DISCHARGE','TOS',200,'corr-0010','2025-10-05 14:20', JSON_OBJECT('crane','QC-02')),
(11,1,'LOAD','TOS',200,'corr-0011','2025-10-06 01:05', JSON_OBJECT('stow','18-04-01')),
(12,2,'DISCHARGE','TOS',200,'corr-0012','2025-10-06 02:30', JSON_OBJECT('crane','QC-07')),
(13,3,'LOAD','TOS',200,'corr-0013','2025-10-06 03:10', JSON_OBJECT('stow','15-12-06')),
(14,4,'GATE_OUT','CMS',200,'corr-0014','2025-10-06 04:00', JSON_OBJECT('truck','SGK5678A')),
(15,5,'LOAD','TOS',200,'corr-0015','2025-10-06 05:45', JSON_OBJECT('stow','09-03-09')),
(16,6,'GATE_IN','CMS',200,'corr-0016','2025-10-06 07:05', JSON_OBJECT('gate','B2')),
(17,7,'LOAD','TOS',200,'corr-0017','2025-10-06 09:20', JSON_OBJECT('stow','03-01-02')),
(18,8,'DISCHARGE','TOS',200,'corr-0018','2025-10-06 12:15', JSON_OBJECT('crane','QC-03')),
(19,9,'LOAD','TOS',200,'corr-0019','2025-10-06 15:40', JSON_OBJECT('stow','05-08-10')),
(20,10,'DISCHARGE','TOS',200,'corr-0020','2025-10-06 16:55', JSON_OBJECT('crane','QC-09'));

-- VIEWS
DROP VIEW IF EXISTS vw_tranship_pipeline;
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

DROP VIEW IF EXISTS vw_edi_last;
CREATE VIEW vw_edi_last AS
SELECT
  c.cntr_no,
  MAX(e.sent_at) AS last_edi_time,
  SUBSTRING_INDEX(GROUP_CONCAT(e.message_type ORDER BY e.sent_at DESC), ',', 1) AS last_edi_type,
  SUBSTRING_INDEX(GROUP_CONCAT(e.status ORDER BY e.sent_at DESC), ',', 1) AS last_edi_status
FROM edi_message e
JOIN container c ON c.container_id = e.container_id
GROUP BY c.cntr_no;


DROP TABLE IF EXISTS berth_application;
DROP TABLE IF EXISTS vessel_advice;

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

-- Seed vessel advice rows
INSERT INTO vessel_advice (vessel_advice_no , vessel_name, system_vessel_name, effective_start_datetime, effective_end_datetime) 
VALUES
 (1000010960,  'MV Lion City 07', 'MV Lion City 07', '2025-10-01 00:00:00', NULL);                  

INSERT INTO vessel_advice (vessel_advice_no, vessel_name, system_vessel_name, effective_start_datetime, effective_end_datetime) VALUES
  (1000010500, 'MV Lion City 08', 'MV Lion City 08', '2025-09-15 00:00:00', '2025-10-01 00:00:00'),
  (1000010961, 'MV Lion City 08', 'MV Lion City 08', '2025-10-01 00:00:00', NULL);

INSERT INTO vessel_advice (vessel_advice_no, vessel_name, system_vessel_name, effective_start_datetime, effective_end_datetime) VALUES
  (1000010962, 'MV Merlion 11', 'MV Merlion 11', '2025-10-02 00:00:00', NULL);

INSERT INTO vessel_advice (vessel_advice_no, vessel_name, system_vessel_name, effective_start_datetime, effective_end_datetime) VALUES
  (1000010400, 'MV Merlion 12', 'MV Merlion 12', '2025-08-01 00:00:00', '2025-09-01 00:00:00'),
  (1000010600, 'MV Merlion 12', 'MV Merlion 12', '2025-09-05 00:00:00', '2025-09-20 00:00:00');

INSERT INTO vessel_advice (vessel_advice_no, vessel_name, system_vessel_name, effective_start_datetime, effective_end_datetime) VALUES
  (1000010700, 'MV Merlion 15', 'MV Merlion 15', '2025-09-10 00:00:00', '2025-09-25 00:00:00'),
  (1000010963, 'MV Merlion 15', 'MV Merlion 15', '2025-09-25 00:00:00', NULL);

INSERT INTO berth_application (vessel_advice_no, vessel_close_datetime, deleted, berthing_status)
VALUES (1000010960, NULL, 'N', 'A');

INSERT INTO berth_application (vessel_advice_no, vessel_close_datetime, deleted, berthing_status) VALUES
  (1000010961, NULL, 'N', 'A'),  
  (1000010962, NULL, 'N', 'A'),   
  (1000010963, NULL, 'N', 'A');   




