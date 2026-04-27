# Capstone Project Plan: Real-Time SCADA Anomaly Detection & Alerting System

## Project Overview

**Title:** Real-Time Industrial Control System Anomaly Detection and Alert Generation Platform

**Description:** A web-based monitoring and alerting system that uses CNN-based anomaly detection to identify cyberattacks in SCADA systems in real-time.

**Core Features:**
1. Real-time sensor data ingestion and monitoring
2. CNN-based anomaly detection (trained model from your research)
3. Automated alert generation when attacks detected
4. Interactive dashboard with visualizations
5. Historical attack logs and analytics

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (Web)                      │
│  - Dashboard  - Real-time Monitoring  - Alerts  - Analytics     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                      BACKEND API (FastAPI/Flask)                 │
│  - Data ingestion  - Model inference  - Alert logic             │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼────────┐
│  ML Pipeline  │  │    Database     │  │  Message Queue │
│ (Prediction)  │  │  (PostgreSQL/   │  │  (Redis/       │
│               │  │   TimescaleDB)  │  │   RabbitMQ)    │
└───────────────┘  └─────────────────┘  └────────────────┘
```

---

## Technology Stack

### Frontend
- **Framework:** React.js or Vue.js
- **UI Library:** Material-UI or Ant Design
- **Charts:** Chart.js or Recharts (for real-time graphs)
- **WebSocket Client:** Socket.IO client (for real-time updates)

### Backend
- **API Framework:** FastAPI (Python) - fast, async, perfect for ML
- **Web Server:** Uvicorn (ASGI)
- **Real-time Communication:** Socket.IO or WebSockets
- **Task Queue:** Celery (for background tasks) or Redis Queue

### Machine Learning
- **Model Serving:** TensorFlow Serving or direct TensorFlow/Keras
- **Preprocessing:** NumPy, Pandas
- **Model Storage:** Saved model files (.h5 or SavedModel format)

### Database
- **Time-series data:** TimescaleDB (PostgreSQL extension) or InfluxDB
- **Alerts/Logs:** PostgreSQL
- **Cache/Queue:** Redis

### DevOps
- **Containerization:** Docker, Docker Compose
- **Deployment:** Cloud (AWS/GCP/Azure) or local server
- **Monitoring:** Prometheus + Grafana (optional, advanced)

---

## Project Phases (8-12 Weeks Timeline)

### Phase 1: Setup & Infrastructure (Week 1-2)

**Objectives:**
- Set up development environment
- Design database schema
- Prepare trained models

**Tasks:**

1. **Environment Setup**
   ```bash
   # Create project structure
   project/
   ├── backend/
   │   ├── api/
   │   ├── models/
   │   ├── services/
   │   └── requirements.txt
   ├── frontend/
   │   ├── src/
   │   └── package.json
   ├── ml_models/
   │   ├── P1_model.h5
   │   ├── P2_model.h5
   │   └── scaler_P1.pkl
   └── docker-compose.yml
   ```

2. **Database Schema Design**
   ```sql
   -- Sensor readings (time-series)
   CREATE TABLE sensor_data (
       id SERIAL PRIMARY KEY,
       timestamp TIMESTAMPTZ NOT NULL,
       stage VARCHAR(10),
       sensor_name VARCHAR(50),
       value FLOAT,
       INDEX idx_timestamp (timestamp DESC)
   );
   
   -- Anomaly detections
   CREATE TABLE anomalies (
       id SERIAL PRIMARY KEY,
       timestamp TIMESTAMPTZ NOT NULL,
       stage VARCHAR(10),
       anomaly_score FLOAT,
       threshold FLOAT,
       is_attack BOOLEAN,
       severity VARCHAR(20),
       details JSONB
   );
   
   -- Alert history
   CREATE TABLE alerts (
       id SERIAL PRIMARY KEY,
       anomaly_id INT REFERENCES anomalies(id),
       created_at TIMESTAMPTZ DEFAULT NOW(),
       alert_type VARCHAR(50),
       message TEXT,
       acknowledged BOOLEAN DEFAULT FALSE,
       acknowledged_at TIMESTAMPTZ,
       acknowledged_by VARCHAR(100)
   );
   ```

3. **Export Trained Models**
   ```python
   # Save each stage model
   for stage in ['P1', 'P2', 'P3', 'P4', 'P5']:
       model.save(f'ml_models/{stage}_model.h5')
       
   # Save scalers
   import pickle
   for stage in ['P1', 'P2', 'P3', 'P4', 'P5']:
       with open(f'ml_models/{stage}_scaler.pkl', 'wb') as f:
           pickle.dump(scaler, f)
   
   # Save statistics (mu, sigma for z-score)
   import json
   stats = {
       'P1': {'mu': mu_P1.tolist(), 'sigma': sigma_P1.tolist()},
       # ... for all stages
   }
   with open('ml_models/stats.json', 'w') as f:
       json.dump(stats, f)
   ```

**Deliverables:**
- ✓ Project repository initialized
- ✓ Database schema created
- ✓ Trained models exported and saved
- ✓ Development environment ready

---

### Phase 2: Backend Development (Week 3-5)

**Objectives:**
- Build REST API
- Implement ML inference pipeline
- Create alert generation logic

**Tasks:**

1. **API Structure (FastAPI)**
   ```python
   # backend/main.py
   from fastapi import FastAPI, WebSocket
   from fastapi.middleware.cors import CORSMiddleware
   
   app = FastAPI(title="SCADA Anomaly Detection API")
   
   # CORS for frontend
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_methods=["*"],
       allow_headers=["*"],
   )
   
   # Routes
   @app.post("/api/ingest")
   async def ingest_data(data: SensorData):
       """Ingest real-time sensor data"""
       pass
   
   @app.get("/api/status")
   async def get_system_status():
       """Get current system status"""
       pass
   
   @app.get("/api/alerts")
   async def get_alerts(skip: int = 0, limit: int = 50):
       """Get recent alerts"""
       pass
   
   @app.post("/api/alerts/{alert_id}/acknowledge")
   async def acknowledge_alert(alert_id: int):
       """Acknowledge an alert"""
       pass
   
   @app.websocket("/ws")
   async def websocket_endpoint(websocket: WebSocket):
       """WebSocket for real-time updates"""
       pass
   ```

2. **ML Inference Service**
   ```python
   # backend/services/anomaly_detector.py
   import numpy as np
   import tensorflow as tf
   from typing import Dict, List
   import pickle
   import json
   
   class AnomalyDetector:
       def __init__(self):
           self.models = {}
           self.scalers = {}
           self.stats = {}
           self.time_steps = 200
           self.buffer = {}  # Store recent data for sequence creation
           
           # Load models
           for stage in ['P1', 'P2', 'P3', 'P4', 'P5']:
               self.models[stage] = tf.keras.models.load_model(
                   f'ml_models/{stage}_model.h5'
               )
               with open(f'ml_models/{stage}_scaler.pkl', 'rb') as f:
                   self.scalers[stage] = pickle.load(f)
           
           # Load statistics
           with open('ml_models/stats.json', 'r') as f:
               self.stats = json.load(f)
           
           # Initialize buffers
           for stage in ['P1', 'P2', 'P3', 'P4', 'P5']:
               self.buffer[stage] = []
       
       def add_data_point(self, stage: str, features: np.ndarray):
           """Add new data point to buffer"""
           self.buffer[stage].append(features)
           
           # Keep only last time_steps + 1 points
           if len(self.buffer[stage]) > self.time_steps + 1:
               self.buffer[stage].pop(0)
       
       def predict(self, stage: str) -> Dict:
           """Run prediction for a stage"""
           if len(self.buffer[stage]) < self.time_steps + 1:
               return {
                   'stage': stage,
                   'status': 'warming_up',
                   'buffer_size': len(self.buffer[stage]),
                   'required': self.time_steps + 1
               }
           
           # Get recent data
           data = np.array(self.buffer[stage][-self.time_steps-1:])
           
           # Scale
           scaled = self.scalers[stage].transform(data)
           
           # Add derivatives
           diff_data = np.diff(scaled, axis=0, prepend=scaled[0].reshape(1, -1))
           enhanced = np.concatenate([scaled, diff_data], axis=1)
           
           # Create sequence
           X = enhanced[:-1].reshape(1, self.time_steps, -1)
           y_true = scaled[-1, :len(scaled[0])//2]  # Original features only
           
           # Predict
           y_pred = self.models[stage].predict(X, verbose=0)[0]
           
           # Calculate error and z-score
           error = np.abs(y_pred - y_true)
           mu = np.array(self.stats[stage]['mu'])
           sigma = np.array(self.stats[stage]['sigma'])
           z_scores = (error - mu) / (sigma + 1e-8)
           max_z = np.max(z_scores)
           
           # Determine if anomaly
           # Use threshold from your grid search results
           threshold = self.get_threshold(stage)  # From your tuning
           is_anomaly = max_z > threshold
           
           return {
               'stage': stage,
               'status': 'ok',
               'max_z_score': float(max_z),
               'threshold': threshold,
               'is_anomaly': is_anomaly,
               'anomaly_score': float(max_z / threshold),  # Normalized score
               'timestamp': None  # Set by caller
           }
       
       def get_threshold(self, stage: str) -> float:
           """Get tuned threshold for stage"""
           # These come from your grid search results
           thresholds = {
               'P1': 3.0,
               'P2': 2.0,
               'P3': 2.0,
               'P4': 2.0,
               'P5': 2.5
           }
           return thresholds.get(stage, 2.5)
   
   # Global detector instance
   detector = AnomalyDetector()
   ```

3. **Alert Generation Service**
   ```python
   # backend/services/alert_service.py
   from datetime import datetime
   from typing import Dict, List
   import asyncio
   
   class AlertService:
       def __init__(self, db_session):
           self.db = db_session
           self.active_alerts = {}
           self.alert_cooldown = 300  # 5 minutes cooldown
       
       async def process_detection(self, detection: Dict):
           """Process anomaly detection and generate alerts if needed"""
           if not detection['is_anomaly']:
               return None
           
           stage = detection['stage']
           
           # Check cooldown to avoid alert spam
           if self.is_in_cooldown(stage):
               return None
           
           # Determine severity
           severity = self.get_severity(detection['anomaly_score'])
           
           # Create alert
           alert = {
               'stage': stage,
               'timestamp': datetime.utcnow(),
               'severity': severity,
               'anomaly_score': detection['anomaly_score'],
               'max_z_score': detection['max_z_score'],
               'threshold': detection['threshold'],
               'message': self.generate_message(stage, severity, detection)
           }
           
           # Save to database
           await self.save_alert(alert)
           
           # Send notifications
           await self.send_notifications(alert)
           
           # Update cooldown
           self.active_alerts[stage] = datetime.utcnow()
           
           return alert
       
       def get_severity(self, anomaly_score: float) -> str:
           """Determine alert severity based on score"""
           if anomaly_score >= 2.0:
               return 'CRITICAL'
           elif anomaly_score >= 1.5:
               return 'HIGH'
           elif anomaly_score >= 1.2:
               return 'MEDIUM'
           else:
               return 'LOW'
       
       def generate_message(self, stage: str, severity: str, detection: Dict) -> str:
           """Generate human-readable alert message"""
           return f"""
           Anomaly Detected in {stage}
           Severity: {severity}
           Anomaly Score: {detection['anomaly_score']:.2f}
           Z-Score: {detection['max_z_score']:.2f} (threshold: {detection['threshold']:.2f})
           
           Possible cyberattack detected. Immediate investigation recommended.
           """
       
       def is_in_cooldown(self, stage: str) -> bool:
           """Check if alert is in cooldown period"""
           if stage not in self.active_alerts:
               return False
           
           last_alert = self.active_alerts[stage]
           elapsed = (datetime.utcnow() - last_alert).total_seconds()
           return elapsed < self.alert_cooldown
       
       async def save_alert(self, alert: Dict):
           """Save alert to database"""
           # Database insert logic
           pass
       
       async def send_notifications(self, alert: Dict):
           """Send notifications via WebSocket, email, etc."""
           # Notification logic
           pass
   ```

4. **Data Ingestion Pipeline**
   ```python
   # backend/services/data_pipeline.py
   from typing import Dict
   import asyncio
   from datetime import datetime
   
   class DataPipeline:
       def __init__(self, detector, alert_service, websocket_manager):
           self.detector = detector
           self.alert_service = alert_service
           self.websocket_manager = websocket_manager
           self.stage_features = self.load_stage_features()
       
       def load_stage_features(self) -> Dict:
           """Load feature mapping for each stage"""
           return {
               'P1': ['FIT 101', 'LIT 101', 'MV 101', 'P1_STATE', 'P101 Status'],
               'P2': ['AIT 201', 'AIT 202', 'AIT 203', 'FIT 201', 'MV201'],
               'P3': ['AIT 301', 'AIT 302', 'DPIT 301', 'FIT 301', 'LIT 301'],
               'P4': ['AIT 402', 'FIT 401', 'LIT 401', 'P401 Status', 'UV401'],
               'P5': ['AIT 501', 'AIT 502', 'FIT 501', 'FIT 502', 'MV 501']
           }
       
       async def process_incoming_data(self, data: Dict):
           """Process incoming sensor data"""
           timestamp = data.get('timestamp', datetime.utcnow())
           
           # Extract features for each stage
           results = []
           
           for stage, features in self.stage_features.items():
               # Extract stage-specific features
               stage_data = np.array([data.get(f, 0.0) for f in features])
               
               # Add to detector buffer
               self.detector.add_data_point(stage, stage_data)
               
               # Run prediction
               detection = self.detector.predict(stage)
               detection['timestamp'] = timestamp
               
               # Process alert if anomaly
               if detection['status'] == 'ok' and detection['is_anomaly']:
                   alert = await self.alert_service.process_detection(detection)
                   if alert:
                       # Broadcast alert via WebSocket
                       await self.websocket_manager.broadcast(alert)
               
               results.append(detection)
           
           # Ensemble detection (OR logic)
           ensemble_anomaly = any(r.get('is_anomaly', False) for r in results)
           
           return {
               'timestamp': timestamp,
               'stages': results,
               'overall_anomaly': ensemble_anomaly
           }
   ```

**Deliverables:**
- ✓ REST API with all endpoints
- ✓ ML inference service running
- ✓ Alert generation logic implemented
- ✓ WebSocket support for real-time updates

---

### Phase 3: Frontend Development (Week 6-8)

**Objectives:**
- Build interactive dashboard
- Implement real-time monitoring
- Create alert management UI

**Tasks:**

1. **Dashboard Layout**
   ```
   ┌─────────────────────────────────────────────────────────────┐
   │  Header: SCADA Anomaly Detection System        [User] [⚙]   │
   ├─────────────────────────────────────────────────────────────┤
   │                                                               │
   │  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐    │
   │  │ Active Alerts │  │ System Status │  │ Detections   │    │
   │  │      5        │  │   🟢 Normal   │  │   Today: 12  │    │
   │  └───────────────┘  └───────────────┘  └──────────────┘    │
   │                                                               │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │        Stage Status Overview                         │   │
   │  │  P1: 🟢  P2: 🟢  P3: 🔴  P4: 🟢  P5: 🟢            │   │
   │  └──────────────────────────────────────────────────────┘   │
   │                                                               │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │        Real-time Sensor Data (Line Charts)           │   │
   │  │  [Live updating graphs for each stage]               │   │
   │  └──────────────────────────────────────────────────────┘   │
   │                                                               │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │        Recent Alerts                                 │   │
   │  │  ⚠️  P3 - CRITICAL - 14:23:45 [Acknowledge] [View]  │   │
   │  │  ⚠️  P1 - HIGH - 14:15:12 [Acknowledge] [View]      │   │
   │  └──────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────┘
   ```

2. **Key React Components**
   ```jsx
   // frontend/src/components/Dashboard.jsx
   import React, { useState, useEffect } from 'react';
   import io from 'socket.io-client';
   import {
     SystemStatus,
     StageMonitor,
     AlertPanel,
     SensorChart
   } from './components';
   
   function Dashboard() {
     const [alerts, setAlerts] = useState([]);
     const [systemStatus, setSystemStatus] = useState({});
     const [sensorData, setSensorData] = useState({});
     const [socket, setSocket] = useState(null);
     
     useEffect(() => {
       // Connect to WebSocket
       const newSocket = io('http://localhost:8000');
       
       newSocket.on('alert', (alert) => {
         setAlerts(prev => [alert, ...prev]);
         // Show notification
         showNotification(alert);
       });
       
       newSocket.on('sensor_update', (data) => {
         setSensorData(prev => ({
           ...prev,
           [data.stage]: [...(prev[data.stage] || []), data]
         }));
       });
       
       setSocket(newSocket);
       
       return () => newSocket.close();
     }, []);
     
     return (
       <div className="dashboard">
         <Header />
         <SystemStatus status={systemStatus} />
         <StageMonitor sensorData={sensorData} />
         <AlertPanel alerts={alerts} />
       </div>
     );
   }
   ```

   ```jsx
   // frontend/src/components/AlertPanel.jsx
   import React from 'react';
   import { Alert, Button, Badge } from 'antd';
   
   function AlertPanel({ alerts }) {
     const getSeverityColor = (severity) => {
       const colors = {
         CRITICAL: 'red',
         HIGH: 'orange',
         MEDIUM: 'yellow',
         LOW: 'blue'
       };
       return colors[severity] || 'default';
     };
     
     const acknowledgeAlert = async (alertId) => {
       await fetch(`/api/alerts/${alertId}/acknowledge`, {
         method: 'POST'
       });
       // Update UI
     };
     
     return (
       <div className="alert-panel">
         <h2>Recent Alerts</h2>
         {alerts.map(alert => (
           <Alert
             key={alert.id}
             type={alert.severity.toLowerCase()}
             message={
               <div>
                 <Badge color={getSeverityColor(alert.severity)} />
                 <strong>{alert.stage}</strong> - {alert.severity}
                 <span style={{float: 'right'}}>
                   {new Date(alert.timestamp).toLocaleString()}
                 </span>
               </div>
             }
             description={alert.message}
             action={
               !alert.acknowledged && (
                 <Button 
                   size="small" 
                   onClick={() => acknowledgeAlert(alert.id)}
                 >
                   Acknowledge
                 </Button>
               )
             }
           />
         ))}
       </div>
     );
   }
   ```

   ```jsx
   // frontend/src/components/SensorChart.jsx
   import React from 'react';
   import { Line } from 'react-chartjs-2';
   
   function SensorChart({ stage, data }) {
     const chartData = {
       labels: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
       datasets: [{
         label: `${stage} Anomaly Score`,
         data: data.map(d => d.anomaly_score),
         borderColor: 'rgb(75, 192, 192)',
         tension: 0.1,
         fill: false
       }]
     };
     
     const options = {
       responsive: true,
       scales: {
         y: {
           beginAtZero: true,
           max: 3
         }
       },
       plugins: {
         annotation: {
           annotations: [{
             type: 'line',
             yMin: data[0]?.threshold || 2.5,
             yMax: data[0]?.threshold || 2.5,
             borderColor: 'red',
             borderWidth: 2,
             label: {
               content: 'Threshold',
               enabled: true
             }
           }]
         }
       }
     };
     
     return <Line data={chartData} options={options} />;
   }
   ```

**Deliverables:**
- ✓ Dashboard UI implemented
- ✓ Real-time data visualization
- ✓ Alert management interface
- ✓ WebSocket integration working

---

### Phase 4: Data Simulation (Week 9)

**Objectives:**
- Create realistic data simulator for demo
- Test system with various scenarios

**Tasks:**

1. **Data Simulator**
   ```python
   # backend/simulator/data_generator.py
   import numpy as np
   import pandas as pd
   import asyncio
   import aiohttp
   from datetime import datetime, timedelta
   
   class SWaTSimulator:
       def __init__(self, api_url="http://localhost:8000"):
           self.api_url = api_url
           self.current_time = datetime.utcnow()
           
           # Load historical data for realistic patterns
           self.historical_data = pd.read_csv('path/to/SWAT.csv')
           self.normal_data = self.historical_data[
               self.historical_data['Attack'] == 0
           ]
           self.attack_data = self.historical_data[
               self.historical_data['Attack'] == 1
           ]
       
       async def simulate_normal_operation(self, duration_minutes=60):
           """Simulate normal operation"""
           async with aiohttp.ClientSession() as session:
               for i in range(duration_minutes * 60):  # 1 sample/second
                   # Sample from normal data
                   sample = self.normal_data.sample(1).iloc[0]
                   
                   # Add some noise
                   data = self.add_noise(sample.to_dict())
                   data['timestamp'] = self.current_time.isoformat()
                   
                   # Send to API
                   await session.post(
                       f'{self.api_url}/api/ingest',
                       json=data
                   )
                   
                   self.current_time += timedelta(seconds=1)
                   await asyncio.sleep(0.1)  # 10x speed
       
       async def simulate_attack(self, attack_type='overflow', duration=300):
           """Simulate specific attack scenario"""
           async with aiohttp.ClientSession() as session:
               for i in range(duration):
                   # Get attack sample or generate synthetic attack
                   sample = self.generate_attack_sample(attack_type, i)
                   sample['timestamp'] = self.current_time.isoformat()
                   
                   await session.post(
                       f'{self.api_url}/api/ingest',
                       json=sample
                   )
                   
                   self.current_time += timedelta(seconds=1)
                   await asyncio.sleep(0.1)
       
       def add_noise(self, sample: dict) -> dict:
           """Add realistic noise to sample"""
           for key, value in sample.items():
               if isinstance(value, (int, float)):
                   # Add 1% Gaussian noise
                   sample[key] = value + np.random.normal(0, abs(value) * 0.01)
           return sample
       
       def generate_attack_sample(self, attack_type: str, step: int) -> dict:
           """Generate synthetic attack data"""
           if attack_type == 'overflow':
               # Simulate tank overflow attack
               sample = self.normal_data.sample(1).iloc[0].to_dict()
               sample['LIT 101'] = 800 + step * 0.5  # Rising water level
               sample['MV 101'] = 1  # Keep valve open
               return sample
           
           elif attack_type == 'sensor_spoof':
               # Fix sensor value
               sample = self.normal_data.sample(1).iloc[0].to_dict()
               sample['LIT 101'] = 700  # Fixed value
               return sample
           
           # Add more attack types...
       
       async def run_demo_scenario(self):
           """Run complete demo with attacks"""
           print("Starting demo scenario...")
           
           # 1. Normal operation (5 minutes)
           print("Phase 1: Normal operation")
           await self.simulate_normal_operation(5)
           
           # 2. Attack 1: Tank overflow (5 minutes)
           print("Phase 2: Tank overflow attack")
           await self.simulate_attack('overflow', 300)
           
           # 3. Recovery (3 minutes)
           print("Phase 3: System recovery")
           await self.simulate_normal_operation(3)
           
           # 4. Attack 2: Sensor spoofing (3 minutes)
           print("Phase 4: Sensor spoofing attack")
           await self.simulate_attack('sensor_spoof', 180)
           
           # 5. Normal operation (5 minutes)
           print("Phase 5: Normal operation resumed")
           await self.simulate_normal_operation(5)
           
           print("Demo completed!")
   
   if __name__ == '__main__':
       simulator = SWaTSimulator()
       asyncio.run(simulator.run_demo_scenario())
   ```

**Deliverables:**
- ✓ Data simulator implemented
- ✓ Attack scenarios defined
- ✓ Demo script ready

---

### Phase 5: Integration & Testing (Week 10)

**Objectives:**
- Integrate all components
- End-to-end testing
- Performance optimization

**Tasks:**

1. **Docker Compose Setup**
   ```yaml
   # docker-compose.yml
   version: '3.8'
   
   services:
     # Database
     postgres:
       image: timescale/timescaledb:latest-pg14
       environment:
         POSTGRES_DB: scada_monitor
         POSTGRES_USER: scada
         POSTGRES_PASSWORD: secure_password
       ports:
         - "5432:5432"
       volumes:
         - postgres_data:/var/lib/postgresql/data
     
     # Redis
     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
     
     # Backend API
     backend:
       build: ./backend
       ports:
         - "8000:8000"
       depends_on:
         - postgres
         - redis
       environment:
         DATABASE_URL: postgresql://scada:secure_password@postgres/scada_monitor
         REDIS_URL: redis://redis:6379
       volumes:
         - ./ml_models:/app/ml_models
       command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
     
     # Frontend
     frontend:
       build: ./frontend
       ports:
         - "3000:3000"
       depends_on:
         - backend
       environment:
         REACT_APP_API_URL: http://localhost:8000
   
   volumes:
     postgres_data:
   ```

2. **Testing Scenarios**
   ```python
   # tests/test_end_to_end.py
   import pytest
   import asyncio
   from httpx import AsyncClient
   
   @pytest.mark.asyncio
   async def test_normal_data_no_alert():
       """Test that normal data doesn't trigger alerts"""
       async with AsyncClient(base_url="http://localhost:8000") as client:
           # Send normal data
           response = await client.post("/api/ingest", json=normal_sample)
           assert response.status_code == 200
           
           # Check no alerts
           alerts = await client.get("/api/alerts")
           assert len(alerts.json()) == 0
   
   @pytest.mark.asyncio
   async def test_attack_generates_alert():
       """Test that attack data triggers alert"""
       async with AsyncClient(base_url="http://localhost:8000") as client:
           # Send attack sequence
           for i in range(250):  # Enough for warm-up + detection
               response = await client.post("/api/ingest", json=attack_sample)
               await asyncio.sleep(0.01)
           
           # Check alert generated
           alerts = await client.get("/api/alerts")
           assert len(alerts.json()) > 0
           assert alerts.json()[0]['severity'] in ['HIGH', 'CRITICAL']
   ```

**Deliverables:**
- ✓ All services containerized
- ✓ Integration tests passing
- ✓ Performance benchmarked

---

### Phase 6: Deployment & Documentation (Week 11-12)

**Objectives:**
- Deploy to cloud/server
- Write documentation
- Prepare presentation

**Tasks:**

1. **Deployment Options**

   **Option A: Cloud (AWS)**
   ```bash
   # Deploy to AWS ECS or EC2
   - Frontend: S3 + CloudFront or Amplify
   - Backend: ECS/Fargate or EC2
   - Database: RDS PostgreSQL with TimescaleDB
   - Redis: ElastiCache
   ```

   **Option B: Local Server**
   ```bash
   # Deploy on university server or local machine
   - Use docker-compose
   - Set up reverse proxy (Nginx)
   - Configure domain/IP access
   ```

2. **Documentation Structure**
   ```
   docs/
   ├── README.md
   ├── ARCHITECTURE.md
   ├── API_DOCUMENTATION.md
   ├── USER_GUIDE.md
   ├── DEPLOYMENT.md
   └── ML_MODEL.md
   ```

3. **Presentation Materials**
   - System architecture diagram
   - Demo video showing:
     * Normal operation
     * Attack detection
     * Alert generation
     * Dashboard interaction
   - Performance metrics
   - Future improvements

**Deliverables:**
- ✓ System deployed and accessible
- ✓ Complete documentation
- ✓ Demo video recorded
- ✓ Presentation slides ready

---

## Key Features Summary

### 1. Real-time Monitoring
- Live sensor data visualization (1 sample/second)
- Stage-by-stage status indicators
- Historical data trending

### 2. Anomaly Detection
- CNN-based detection (your trained models)
- Per-stage analysis (P1-P5)
- Ensemble decision making

### 3. Alert System
- Automatic alert generation
- Severity classification (CRITICAL/HIGH/MEDIUM/LOW)
- Alert cooldown to prevent spam
- Acknowledgement workflow

### 4. Dashboard
- Interactive web interface
- Real-time updates via WebSocket
- Alert history and management
- System analytics

### 5. Simulation
- Realistic data generation for demo
- Multiple attack scenarios
- Controllable timing and intensity

---

## Evaluation Criteria (Capstone)

### Technical Implementation (40%)
- ✓ Working ML model integration
- ✓ Real-time data pipeline
- ✓ Database design
- ✓ API implementation
- ✓ Frontend functionality

### Innovation & Complexity (25%)
- ✓ CNN-based anomaly detection
- ✓ Real-time processing
- ✓ Ensemble approach
- ✓ Advanced visualization

### Documentation (20%)
- ✓ System architecture
- ✓ API documentation
- ✓ User guide
- ✓ Code comments

### Demonstration (15%)
- ✓ Live demo or video
- ✓ Multiple scenarios
- ✓ Clear explanation
- ✓ Q&A handling

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|-----------------|
| 1-2 | Setup & Infrastructure | Database, models exported |
| 3-5 | Backend Development | API, ML inference, alerts |
| 6-8 | Frontend Development | Dashboard, real-time UI |
| 9 | Data Simulation | Simulator, attack scenarios |
| 10 | Integration & Testing | Docker, tests, optimization |
| 11-12 | Deployment & Docs | Deployment, documentation, demo |

---

## Risk Mitigation

### Technical Risks
1. **Real-time performance issues**
   - Mitigation: Use async processing, Redis caching, optimize model inference
   
2. **Model accuracy in production**
   - Mitigation: Extensive testing, adjustable thresholds, manual override

3. **WebSocket connection stability**
   - Mitigation: Reconnection logic, fallback to polling

### Project Risks
1. **Scope creep**
   - Mitigation: Stick to MVP, document future features separately

2. **Time constraints**
   - Mitigation: Prioritize core features, use libraries/frameworks

---

## Success Metrics

✓ **System can process** 1 sample/second in real-time
✓ **Detection accuracy** ≥ 85% F1 score (matching your research)
✓ **Alert latency** < 30 seconds from attack start
✓ **Dashboard responsive** < 2 second load time
✓ **99% uptime** during demo period
✓ **Complete documentation** for all components

---

## Next Steps

1. **Review and approve** this plan with advisor
2. **Set up** development environment (Week 1)
3. **Start with** Phase 1 tasks
4. **Weekly progress** reviews
5. **Adjust timeline** as needed

Good luck with your capstone project! 🚀
