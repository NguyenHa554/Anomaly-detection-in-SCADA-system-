# SCADA Anomaly Detection System

A real-time industrial control system anomaly detection and alerting platform using CNN-based machine learning models trained on the SWaT (Secure Water Treatment) dataset.

## Overview

This system monitors SCADA (Supervisory Control and Data Acquisition) infrastructure in real-time, detecting cyberattacks and anomalies across multiple processing stages (P1-P6). It uses deep learning models to identify abnormal patterns in sensor data and generates alerts when attacks are detected.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                   │
│              Dashboard │ Alerts │ Real-time Charts           │
└────────────────────────────┬────────────────────────────────┘
                             │ WebSocket + REST
┌────────────────────────────┼────────────────────────────────┐
│                     Backend (FastAPI)                       │
│    API │ Anomaly Detection │ Alert Service │ Data Pipeline   │
└────────────────────────────┬────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
   ┌─────▼─────┐    ┌────────▼────────┐   ┌────▼────┐
   │   ML      │    │   SQLite DB    │   │ Simula- │
   │  Models   │    │  (Alerts/Data) │   │  tor    │
   └───────────┘    └─────────────────┘   └─────────┘
```

## Project Structure

```
├── backend/                  # FastAPI backend
│   ├── main.py              # API entry point
│   ├── database.py          # SQLite database models
│   ├── simulator.py         # Data simulator
│   ├── requirements.txt     # Python dependencies
│   └── services/
│       ├── anomaly_detector.py   # ML inference
│       ├── data_pipeline.py      # Data processing
│       └── alert_service.py      # Alert generation
│
├── frontend/                 # React frontend (Vite)
│   └── src/                 # React components
│
├── ml_models/               # Trained CNN models
│   ├── P2_model.keras      # Stage P2 model
│   ├── P3_model.keras      # Stage P3 model
│   ├── P4_model.keras      # Stage P4 model
│   ├── P5_model.keras      # Stage P5 model
│   ├── P6_model.keras      # Stage P6 model
│   ├── *_scaler.pkl         # Feature scalers
│   ├── stats.json          # Z-score statistics
│   └── thresholds.json     # Detection thresholds
│
├── data2.ipynb              # Model training notebook
├── export_models.py         # Model export script
└── SWaT.csv                 # Dataset
```

## Features

- **Real-time Monitoring**: Live sensor data ingestion and visualization
- **Multi-stage Detection**: Anomaly detection for P2-P6 processing stages
- **CNN-based Models**: Deep learning models using Convolutional Neural Networks
- **Automatic Alerts**: Severity-classified alerts (CRITICAL/HIGH/MEDIUM/LOW)
- **WebSocket Support**: Real-time updates to dashboard
- **Alert Management**: Acknowledge and track alerts
- **Data Simulation**: Built-in simulator for testing and demos

## Technology Stack

### Backend
- **FastAPI** - Async REST API framework
- **Uvicorn** - ASGI server
- **TensorFlow/Keras** - ML model inference
- **SQLAlchemy** - Database ORM
- **NumPy/Pandas** - Data processing

### Frontend
- **React** - UI framework
- **Vite** - Build tool
- **Chart.js** - Real-time charts

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Model Setup

If models are not exported, run:

```bash
python export_models.py
```

This will create the `ml_models/` directory with all trained models, scalers, and configuration files.

### Frontend Setup

```bash
cd frontend
npm install
```

## Running the Application

### Start Backend

```bash
cd backend
uvicorn backend.main:app --reload --port 8000
```

### Start Frontend

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/api/status` | GET | System status and model info |
| `/api/ingest` | POST | Ingest sensor data |
| `/api/alerts` | GET | Get alert history |
| `/api/alerts/{id}/acknowledge` | POST | Acknowledge an alert |
| `/api/history` | GET | Get sensor reading history |
| `/api/runtime/reload` | POST | Reload models |
| `/api/runtime/reset` | POST | Reset runtime state |
| `/ws` | WebSocket | Real-time updates |

## Detection Stages

The system monitors 5 production stages:

| Stage | Features | Description |
|-------|----------|-------------|
| P1 | AIT 101, FIT 101, MV101 | Raw Water |
| P2 | AIT 201-203, FIT 201, MV201 | Chemical dosing |
| P3 | AIT 301-303, DPIT 301, FIT 301, LIT 301 | Dechlorination |
| P4 | AIT 401-402, FIT 401, LIT 401, UV401 | UV disinfection |
| P5 | AIT 501-504, FIT 501-504, PIT 501-503 | Filtration |
| P6 | FIT 601, LSH 601-603, LSL 601-603 | Discharge |

## Data Format

Sensor data should be sent as JSON with column names matching SWaT dataset:

```json
{
  "AIT 201": 120.5,
  "AIT 202": 118.2,
  "FIT 201": 45.3,
  ...
}
```

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend
cd frontend
npm run lint
```

### Docker (Optional)

```bash
docker-compose up --build
<<<<<<< HEAD
```
=======

```
### Running simulator 
$env:SIM_START_ROW=''
$env:SIM_END_ROW=''
$env:SIM_DELAY=''
python -m backend.simulator
>>>>>>> 717f6ba (update threshold UI)

## SWaT Dataset

This system is trained on the Secure Water Treatment (SWaT) dataset, a widely-used benchmark for industrial control system security research. The dataset contains 11 days of operational data with 25 attacks.

## License

This project is for educational/research purposes.

## References

- SWaT Dataset: https://itrust.sutd.edu.sg/itrust-labs-home datasets/swat/
<<<<<<< HEAD
- Related research papers on CNN-based anomaly detection for SCADA systems
=======
- Related research papers on CNN-based anomaly detection for SCADA systems
>>>>>>> 717f6ba (update threshold UI)
