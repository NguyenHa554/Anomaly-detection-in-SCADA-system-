# Dataset Info Sheet A.12

## Dataset Name
**SWaT.A12_Mar 2026**

---

## Description
This dataset contains network traffic and historian data captured and/or collected on 11 March 2026, from the SWaT Testbed.

The SWaT Testbed was run for a duration of 8 hours:
- The first 4 hours consist of normal testbed run data
- The latter 4 hours consist of attacks carried out during a testbed run

The purpose of the dataset is to provide:
- A baseline dataset
- An attack dataset of SWaT operations

This dataset is intended for analysis of ICS network data and was collected as per the request of Ensign in preparation for a User Acceptance Test (Pre-UAT) to test detection systems.

---

## Data Source

### PCAP file
- Captured by an aggregator device installed on the water treatment plant network
- Aggregates traffic from multiple network segments
- Stored in PCAP format

### CSV file
- Generated using a custom Python script
- Extracted from the Historian
- Contains tag values

---

## Data Volume

### PCAP file
- Size: **7.67 GB (7.68 GB)**
- Total files: **179**

### CSV file
- Size: **15.7 MB**
- Records: **28,861**
- Time range: **09:00:00 → 17:00:59**

---

## Dataset Structure

| # | Folder Name                     | Description                      |
|--|--------------------------------|----------------------------------|
| 1 | SWaT.A12_OTDataset_Mar_26     | SWaT OT datasets in CSV format  |
| 2 | SWaT.A12_PCAPs_Mar_26         | SWaT Wireshark PCAP files       |

---

## Data Fields

### PCAP file
- **Timestamp**: Date and time when packet was captured
- **Source IP**: Sender IP address
- **Destination IP**: Receiver IP address
- **Protocol**: Network protocol (e.g., TCP, UDP)
- **Length**: Packet size in bytes
- **Description**: Brief content description

### CSV file
- **Timestamp**: State of SWaT testbed per second
- **Process States**: e.g., `P1_STATE`
- **Actuator & Sensor States**: e.g., `MV101.Status`, `FIT101.Pv`
- **Alarm Statuses**: e.g., `LSL603.Alarm`

---

## Attacks (Overview)

- **Normal Testbed Run**: 4 hours
- **Attack Testbed Run**: 4 hours

---

## Attack Scenarios

| # | Attack Name                                                                 | Start Time | End Time | Primary Targets              |
|--|----------------------------------------------------------------------------|------------|----------|------------------------------|
| 1 | Stage 5 Valve Manipulation — MV503/504 Open, MV501/502 Close               | 1:00:00 PM | 1:05:00 PM | MV501–504                   |
| 2 | Stage 1 Flow Disruption — MV101 Open, P101 & P102 Stop                     | 1:40:00 PM | 1:45:00 PM | MV101, P101, P102           |
| 3 | Florida Water Plant Scenario — Dosing Pump Activation & Sensor Spoofing    | 2:20:00 PM | 2:25:00 PM | P201–P206, MV201            |
| 4 | Tank Overflow via LIT101 Spoofing — Schneider Demo Attack 01               | 2:30:00 PM | 2:35:00 PM | LIT101                      |
| 5 | Stage 5 Valve Manipulation — MV503/504 Open, MV501/502 Close (Repeat)      | 2:40:00 PM | 2:45:00 PM | MV501–504                   |
| 6 | Tank Overflow via LIT101 Spoofing — Schneider Demo Attack 01               | 3:00:00 PM | 3:02:00 PM | LIT101                      |
| 7 | Stage 2 Parallel Pump Override — MV201 Open, P101 & P102 Run               | 3:02:00 PM | 3:07:00 PM | MV201, P101, P102           |
| 8 | Reverse Osmosis Backwash Diversion — MV302 Close, MV303 Open               | 3:20:00 PM | 3:25:00 PM | MV302, MV303                |
| 9 | Forced Backwash Trigger via DPIT301 Spoofing — Ensign Pre-UAT Attack 1     | 3:45:00 PM | 3:50:00 PM | DPIT301                     |
|10 | Multi-Value Level Oscillation — LIT601 Spoofing Sequence — Ensign Pre-UAT Attack 3 | 4:10:00 PM | 4:15:00 PM | LIT601                      |
|11 | AIT402 High-Value Spoof Start — Using script                               | 4:35:00 PM | 4:40:00 PM | AIT402                      |
