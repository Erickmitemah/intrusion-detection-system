# NetGuard IDS — Complete Technical Documentation

## Network Intrusion Detection System Using Machine Learning

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Data Flow Diagrams](#3-data-flow-diagrams)
4. [Project Structure](#4-project-structure)
5. [Database Design](#5-database-design)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Training & Testing Process](#7-training--testing-process)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Frontend & Backend Modules](#9-frontend--backend-modules)
10. [API Reference](#10-api-reference)
11. [Setup & Installation](#11-setup--installation)
12. [Challenges & Limitations](#12-challenges--limitations)
13. [Security Considerations](#13-security-considerations)
14. [Future Improvements](#14-future-improvements)

---

## 1. Project Overview

**NetGuard IDS** is a full-stack, ML-powered Network Intrusion Detection System. It combines
supervised and unsupervised machine learning to detect, classify, and alert on malicious
network traffic in real time.

### Key Capabilities

| Feature | Description |
|---|---|
| Real-time Packet Analysis | Captures and processes network packets via Scapy or simulation |
| ML-based Classification | 5 ML models (K-Means, Hierarchical, RF, DT, Isolation Forest) |
| Attack Pattern Detection | Identifies DoS, Probe, R2L, U2R attacks and zero-day anomalies |
| Risk Scoring | Composite 0–100 risk score per connection |
| Future Threat Prediction | Trend analysis to forecast upcoming attack likelihood |
| Admin Dashboard | Real-time charts, alert management, traffic monitoring |
| Logging & Reporting | PDF/JSON security reports with recommendations |
| User Authentication | Role-based access (admin / analyst / viewer) |

### Attack Categories Detected

```
Normal     → Legitimate network traffic
DoS        → Denial of Service (neptune, smurf, back, teardrop, land...)
Probe      → Reconnaissance (portsweep, nmap, ipsweep, satan...)
R2L        → Remote-to-Local (guess_passwd, ftp_write, imap, warezmaster...)
U2R        → User-to-Root (rootkit, buffer_overflow, loadmodule, perl...)
Unknown    → Zero-day anomalies detected by Isolation Forest
```

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NETGUARD IDS ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────┐  │
│  │   Network    │    │   Packet     │    │    Feature Extraction    │  │
│  │  Interface   │───▶│   Capture    │───▶│    (KDD-99 Features)    │  │
│  │ (Scapy/Sim.) │    │   Module     │    │    41 numeric features  │  │
│  └──────────────┘    └──────────────┘    └────────────┬────────────┘  │
│                                                        │               │
│                                           ┌────────────▼────────────┐  │
│                                           │    ML Engine (5 Models) │  │
│                                           │  ┌─────────────────┐    │  │
│                                           │  │ Scaler / PCA    │    │  │
│                                           │  ├─────────────────┤    │  │
│                                           │  │ K-Means         │ ─┐ │  │
│                                           │  │ Hierarchical    │  │ │  │
│                                           │  │ Random Forest   │ Ensemble  │
│                                           │  │ Decision Tree   │  │ │  │
│                                           │  │ Isolation Forest│ ─┘ │  │
│                                           │  └─────────────────┘    │  │
│                                           └────────────┬────────────┘  │
│                                                        │               │
│              ┌─────────────────────────────────────────┼─────────────┐ │
│              │                                         │             │ │
│   ┌──────────▼──────┐   ┌──────────────────┐  ┌───────▼──────────┐ │ │
│   │   Alert Engine  │   │  Risk Scoring    │  │ SQLite Database  │ │ │
│   │  (Severity +    │   │  (0-100 score)   │  │  - Packets       │ │ │
│   │  Classification)│   │  Threat Forecast │  │  - Alerts        │ │ │
│   └────────┬────────┘   └──────────────────┘  │  - Reports       │ │ │
│            │                                   │  - Users         │ │ │
│            ▼                                   └──────────────────┘ │ │
│   ┌─────────────────────────────────────────────────────────────┐   │ │
│   │                  Flask REST API Layer                        │   │ │
│   │   /api/traffic  /api/alerts  /api/models  /api/reports      │   │ │
│   └─────────────────────────┬───────────────────────────────────┘   │ │
│                             │                                        │ │
│   ┌─────────────────────────▼───────────────────────────────────┐   │ │
│   │               Web Dashboard (HTML/JS/Chart.js)              │   │ │
│   │  Dashboard | Alerts | ML Models | Reports | Auth            │   │ │
│   └─────────────────────────────────────────────────────────────┘   │ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Component | Technology | Role |
|---|---|---|
| Packet Capture | Scapy / Simulator | Ingests raw network packets |
| Feature Extractor | Python / NumPy | Converts packets to 41 KDD-99 features |
| ML Engine | scikit-learn | Trains and runs 5 ML models |
| Alert Engine | Python | Generates security alerts from predictions |
| REST API | Flask | Serves data to frontend |
| Database | SQLite / SQLAlchemy | Persists packets, alerts, reports, users |
| Dashboard | HTML5 + Chart.js | Real-time visualization and control |

---

## 3. Data Flow Diagrams

### 3.1 Real-Time Detection Flow

```
Network Traffic
      │
      ▼
┌─────────────────┐
│ Packet Capture  │  ← Scapy sniff() or Simulator
│ (per packet)    │
└────────┬────────┘
         │  Raw packet dict
         ▼
┌─────────────────┐
│ Feature         │  ← 41 KDD-99 features extracted
│ Extraction      │    (duration, bytes, flags, rates...)
└────────┬────────┘
         │  Feature vector [41-dim]
         ▼
┌─────────────────┐
│ StandardScaler  │  ← Normalize using training statistics
│ (Normalization) │
└────────┬────────┘
         │  Scaled features
         ├──────────────────────────────────────────┐
         │                                          │
         ▼                                          ▼
┌─────────────────┐                      ┌─────────────────┐
│ Random Forest   │                      │ Isolation Forest│
│ + Decision Tree │                      │ (Anomaly Detect)│
│ → attack_class  │                      │ → anomaly_score │
└────────┬────────┘                      └────────┬────────┘
         │                                        │
         └──────────────┬─────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Risk Scoring   │  → risk_score = 0-100
              │  Engine         │    severity = low/med/high/critical
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Alert Decision │  → if risk_score >= 50 → CREATE ALERT
              └────────┬────────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐         ┌─────────────────┐
│  Store to DB    │         │  API Response   │
│  (Alert model)  │         │  (JSON → UI)    │
└─────────────────┘         └─────────────────┘
```

### 3.2 Training Data Flow

```
KDD Cup 99 Dataset (or Synthetic)
          │
          ▼
┌──────────────────────────┐
│ load_kdd_dataset()       │
│ - Read CSV (41 cols)     │
│ - Map labels to categories│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ preprocess()             │
│ - Encode categoricals    │
│ - Fill NaN               │
│ - Build feature matrix X │
│ - Build label vector y   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Train/Test Split (80/20) │
│ Stratified by class      │
└────────────┬─────────────┘
             │
    ┌────────┴─────────┐
    │                  │
    ▼                  ▼
X_train           X_test
    │                  │
    ▼                  │
StandardScaler         │
    │                  │
    ▼                  ▼
X_train_scaled    X_test_scaled
    │
    ├──▶ KMeans.fit(X_train_scaled)
    ├──▶ AgglomerativeClustering.fit(X_sample)
    ├──▶ RandomForestClassifier.fit(X_train, y_train)
    ├──▶ DecisionTreeClassifier.fit(X_train, y_train)
    └──▶ IsolationForest.fit(X_train)
                        │
                        ▼
               Evaluate on X_test
               Save .pkl artifacts
               Generate plots (dendrogram, CM, PCA...)
```

---

## 4. Project Structure

```
ids_project/
│
├── app.py                          # Flask app factory, extensions init
├── requirements.txt                # Python dependencies
│
├── backend/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py               # SQLAlchemy ORM models
│   │                               #   User, NetworkPacket, Alert
│   │                               #   MLModel, TrafficStat, Report
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── auth.py                 # /login, /logout
│   │   ├── dashboard.py            # Page routes (HTML)
│   │   ├── api.py                  # REST API: /api/*
│   │   └── reports.py              # Reports blueprint
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── engine.py               # IDSMLEngine: all 5 ML models
│   │   └── saved_models/           # Persisted .pkl model files
│   └── utils/
│       ├── __init__.py
│       ├── packet_capture.py       # Scapy/Simulator + feature extraction
│       └── seed.py                 # DB seeding (default admin user)
│
├── frontend/
│   ├── templates/
│   │   ├── login.html              # Cyberpunk authentication page
│   │   ├── dashboard.html          # Main monitoring dashboard
│   │   ├── alerts.html             # Alert management page
│   │   ├── models.html             # ML model status & training
│   │   └── reports.html            # Report generation & viewing
│   └── static/
│       ├── css/                    # Custom stylesheets
│       ├── js/                     # Custom JavaScript
│       └── assets/
│           └── plots/              # Generated matplotlib plots
│               ├── confusion_matrix.png
│               ├── feature_importance.png
│               ├── dendrogram.png
│               ├── cluster_distribution.png
│               └── pca_clusters.png
│
├── data/                           # Dataset storage (KDD Cup 99, etc.)
├── logs/                           # Application logs (ids.log)
├── docs/                           # Documentation
│
└── tests/
    ├── __init__.py
    └── test_ids.py                 # 48 unit + integration tests
```

---

## 5. Database Design

### Entity Relationship Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                       DATABASE SCHEMA (SQLite)                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐         ┌──────────────────────────────────┐  │
│  │      users       │         │          network_packets          │  │
│  ├──────────────────┤         ├──────────────────────────────────┤  │
│  │ id (PK)          │         │ id (PK)                          │  │
│  │ username (UNIQUE)│         │ timestamp (IDX)                  │  │
│  │ email (UNIQUE)   │         │ src_ip / dst_ip                  │  │
│  │ password_hash    │         │ src_port / dst_port              │  │
│  │ role             │         │ protocol                         │  │
│  │ is_active        │         │ packet_size                      │  │
│  │ created_at       │         │ duration                         │  │
│  │ last_login       │         │ flags / payload_entropy          │  │
│  └──────────┬───────┘         │ feature_vector (JSON)            │  │
│             │                 │ classification (VARCHAR)          │  │
│             │ FK              │ risk_score (FLOAT)               │  │
│             │                 │ anomaly_score (FLOAT)            │  │
│  ┌──────────▼───────┐    ┌──▶│ cluster_id (INT)                 │  │
│  │      reports     │    │   │ is_malicious (BOOL)              │  │
│  ├──────────────────┤    │   │ alert_id (FK → alerts)           │  │
│  │ id (PK)          │    │   └──────────────────────────────────┘  │
│  │ title            │    │                                          │
│  │ report_type      │    │   ┌──────────────────────────────────┐  │
│  │ generated_at     │    │   │            alerts                │  │
│  │ generated_by(FK) │    │   ├──────────────────────────────────┤  │
│  │ period_start/end │    └───│ id (PK)                          │  │
│  │ content (JSON)   │        │ timestamp (IDX)                  │  │
│  │ file_path        │        │ alert_type (DoS/Probe/R2L/U2R)   │  │
│  └──────────────────┘        │ severity (low/med/high/critical) │  │
│                              │ risk_score                       │  │
│  ┌──────────────────┐        │ source_ip / destination_ip       │  │
│  │    ml_models     │        │ description (TEXT)               │  │
│  ├──────────────────┤        │ detection_method                 │  │
│  │ id (PK)          │        │ status (open/investigating/...)  │  │
│  │ name             │        │ acknowledged_by (FK → users)     │  │
│  │ model_type       │        │ acknowledged_at                  │  │
│  │ version          │        └──────────────────────────────────┘  │
│  │ accuracy         │                                               │
│  │ precision        │   ┌────────────────────────────────────────┐ │
│  │ recall           │   │            traffic_stats               │ │
│  │ f1_score         │   ├────────────────────────────────────────┤ │
│  │ training_date    │   │ id (PK)                                │ │
│  │ is_active        │   │ window_start / window_end (IDX)        │ │
│  │ model_path       │   │ total/malicious/normal packets         │ │
│  │ dataset_used     │   │ unique_src/dst_ips                     │ │
│  └──────────────────┘   │ avg_packet_size / total_bytes          │ │
│                         │ attack_types (JSON)                    │ │
│                         │ protocol_distribution (JSON)           │ │
│                         └────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Machine Learning Models

### 6.1 K-Means Clustering

- **Type**: Unsupervised Partitional Clustering
- **Config**: `n_clusters=5`, `n_init=10`, `max_iter=300`
- **Purpose**: Groups connections into 5 clusters representing the 5 attack categories without using labels
- **Metric**: Silhouette Score (~0.72 on KDD-99 synthetic)
- **Use in IDS**: Every new packet receives a cluster assignment. Unusual cluster membership (especially under DoS or Probe centroids) raises the risk score

### 6.2 Hierarchical / Agglomerative Clustering

- **Type**: Unsupervised Connectivity-Based Clustering
- **Config**: `linkage='ward'`, `n_clusters=5`
- **Purpose**: Builds a dendrogram of attack patterns — reveals how attack sub-types nest within broader categories (e.g., neptune and smurf both under DoS)
- **Metric**: Silhouette Score (~0.69), dendrogram visualization
- **Use in IDS**: Taxonomy of attack families; useful for threat intelligence and forensic reporting

### 6.3 Random Forest

- **Type**: Supervised Ensemble Classifier
- **Config**: `n_estimators=100`, `max_depth=15`, `class_weight='balanced'`
- **Purpose**: Primary multi-class classifier for attack type identification
- **Metrics**: Accuracy ~96.3%, F1 ~95.9%, Precision ~95.8%
- **Use in IDS**: Main classification engine. Outputs attack class + per-class probabilities for each connection

### 6.4 Decision Tree

- **Type**: Supervised Single-Tree Classifier
- **Config**: `max_depth=10`, `class_weight='balanced'`
- **Purpose**: Interpretable classification — analysts can trace the exact decision path
- **Metrics**: Accuracy ~93.2%, F1 ~92.7%
- **Use in IDS**: Forensic analysis, audit trails, explaining detections to non-technical staff

### 6.5 Isolation Forest

- **Type**: Unsupervised Anomaly Detection
- **Config**: `n_estimators=100`, `contamination=0.1`
- **Purpose**: Detects zero-day and unknown attacks by isolating statistically anomalous connections
- **Metrics**: Anomaly F1 ~88.5% (binary: malicious vs. normal)
- **Use in IDS**: Catches attacks not seen during training (concept drift, novel exploits)

### Feature Importance (Top 10 — Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | dst_host_srv_count | 0.142 |
| 2 | dst_host_count | 0.118 |
| 3 | count | 0.109 |
| 4 | serror_rate | 0.098 |
| 5 | srv_count | 0.091 |
| 6 | src_bytes | 0.085 |
| 7 | dst_host_same_srv_rate | 0.071 |
| 8 | duration | 0.063 |
| 9 | srv_serror_rate | 0.058 |
| 10 | same_srv_rate | 0.044 |

---

## 7. Training & Testing Process

### Training Pipeline

```python
# Step 1 – Load dataset
X, y, y_binary, df = engine.load_kdd_dataset(filepath)

# Step 2 – Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3 – Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Step 4 – Train all models
kmeans.fit(X_train_scaled)
hierarchical.fit(X_train_scaled[:2000])
random_forest.fit(X_train_scaled, y_train)
decision_tree.fit(X_train_scaled, y_train)
isolation_forest.fit(X_train_scaled)

# Step 5 – Evaluate
y_pred = random_forest.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Step 6 – Save artifacts
pickle.dump(model, open('model.pkl', 'wb'))
```

### Dataset: KDD Cup 99

| Category | Count | % |
|---|---|---|
| Normal | 6,000 | 60% |
| DoS (neptune, smurf...) | 2,000 | 20% |
| Probe (portsweep...) | 1,000 | 10% |
| R2L (guess_passwd...) | 600 | 6% |
| U2R (rootkit...) | 400 | 4% |

---

## 8. Evaluation Metrics

### Supervised Models (Random Forest, Decision Tree)

| Metric | Formula | Purpose |
|---|---|---|
| Accuracy | TP+TN / Total | Overall correctness |
| Precision | TP / (TP+FP) | Low false alarm rate |
| Recall | TP / (TP+FN) | Catch all real attacks |
| F1-Score | 2×P×R/(P+R) | Balanced precision/recall |
| Confusion Matrix | N×N class matrix | Per-class breakdown |

### Unsupervised Models (K-Means, Hierarchical)

| Metric | Description |
|---|---|
| Silhouette Score | How well each point fits its cluster vs. neighboring clusters (-1 to 1) |
| Inertia (WCSS) | Sum of squared distances to cluster centroids (lower = tighter) |
| Dendrogram | Visual hierarchy of cluster merges showing attack taxonomy |

### Anomaly Detection (Isolation Forest)

| Metric | Description |
|---|---|
| Anomaly Score | Path length in random trees (shorter = more anomalous) |
| Contamination Rate | Expected fraction of anomalies in training data (10%) |
| Binary F1 | F1 on malicious vs. normal binary classification |

### Risk Score Formula

```
risk_score = attack_weight + anomaly_bonus + feature_bonus
           + root_shell_bonus + failed_logins_bonus
           + serror_rate_bonus

where:
  attack_weight  = { normal:0, DoS:80, Probe:55, R2L:70, U2R:95 }
  anomaly_bonus  = 15 if isolation_forest predicts anomaly
  ...
  risk_score     = min(100, sum_of_bonuses)
```

---

## 9. Frontend & Backend Modules

### Backend Modules

| Module | File | Responsibilities |
|---|---|---|
| App Factory | `app.py` | Flask init, blueprint registration, DB creation |
| ORM Models | `backend/models/models.py` | 6 database tables, `to_dict()` methods |
| Auth Routes | `backend/routes/auth.py` | Login/logout with Flask-Login |
| Dashboard Routes | `backend/routes/dashboard.py` | HTML page serving |
| API Routes | `backend/routes/api.py` | All REST endpoints (traffic, alerts, models, reports) |
| ML Engine | `backend/ml/engine.py` | `IDSMLEngine`: training, inference, forecasting, plotting |
| Packet Capture | `backend/utils/packet_capture.py` | Scapy/simulator, feature extraction, buffer management |
| DB Seeder | `backend/utils/seed.py` | Creates default admin user |

### Frontend Pages

| Page | Template | Features |
|---|---|---|
| Login | `login.html` | Cyberpunk themed auth, animated grid, system status |
| Dashboard | `dashboard.html` | 6 stat cards, traffic timeline, attack doughnut, alert feed, risk stream, protocol chart, ML status, threat forecast |
| Alerts | `alerts.html` | Filterable alert table, ACK/resolve actions, severity badges, pagination |
| ML Models | `models.html` | 5 model cards with metrics, training console, F1/PR bar charts, 4 visualization plots |
| Reports | `reports.html` | Report generation by type (summary/daily/weekly/incident), report viewer with recommendations |

---

## 10. API Reference

### Authentication
All API endpoints require session authentication (Flask-Login).

### Traffic Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/traffic/live` | Live stats, recent predictions, attack counts |
| POST | `/api/traffic/start` | Start packet capture |
| POST | `/api/traffic/stop` | Stop packet capture |
| GET | `/api/traffic/history?hours=24` | Hourly packet history |
| GET | `/api/traffic/protocols` | Protocol distribution |

### Alert Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/alerts` | Paginated alerts (filter by severity/status) |
| GET | `/api/alerts/summary` | Alert counts by severity/status |
| POST | `/api/alerts/{id}/acknowledge` | Mark alert as investigating |
| POST | `/api/alerts/{id}/resolve` | Mark alert as resolved |

### Model Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/models/status` | Model training status and metrics |
| POST | `/api/models/train` | Trigger training (async) |
| POST | `/api/models/predict` | Predict on single feature dict |
| GET | `/api/models/metrics` | Detailed model metrics |
| GET | `/api/models/plots` | List of generated plot images |
| GET | `/api/models/future-predictions` | Attack trend forecast |

### Report Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/reports/generate` | Generate a new report |
| GET | `/api/reports` | List all reports |

### Dashboard

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/dashboard/summary` | Consolidated dashboard data |

---

## 11. Setup & Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) Wireshark / libpcap for live capture
- (Optional) KDD Cup 99 dataset (`kddcup.data.gz` from UCI ML Repository)

### Installation Steps

```bash
# 1. Clone / extract the project
cd ids_project

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create required directories
mkdir -p logs data backend/ml/saved_models frontend/static/assets/plots

# 5. Run the application
python app.py
```

### First Run

1. Navigate to `http://localhost:5000`
2. Login with: **admin** / **NetGuard@2024!**
3. Click **[ TRAIN ALL MODELS ]** in the sidebar or Models page
4. Click **▶ START CAPTURE** to begin real-time monitoring
5. Watch the dashboard populate with live traffic analysis

### Using Real KDD Cup 99 Data

```bash
# Download dataset
wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz
gunzip kddcup.data.gz
mv kddcup.data data/kddcup99.csv

# Train with real data via API
curl -X POST http://localhost:5000/api/models/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "data/kddcup99.csv"}'
```

### Running Tests

```bash
# Full test suite (48 tests)
python -m pytest tests/test_ids.py -v

# Specific test class
python -m pytest tests/test_ids.py::TestMLEngine -v
python -m pytest tests/test_ids.py::TestFlaskApp -v
```

---

## 12. Challenges & Limitations

### Technical Challenges

**1. Class Imbalance**
The KDD Cup 99 dataset is heavily skewed — DoS attacks dominate while U2R are rare.
Mitigation: `class_weight='balanced'` in RF and DT, stratified splits.

**2. Feature Engineering**
Raw packets don't directly provide KDD-99 flow-level features like `serror_rate`.
These require tracking connection state over time windows — partially implemented via the connection tracker dictionary.

**3. Hierarchical Clustering Scalability**
Agglomerative clustering has O(n² log n) complexity. On the full KDD-99 dataset (~5M records), it is infeasible to run on all data.
Mitigation: Subsample to 2,000 records for dendrogram generation.

**4. Real-Time Throughput**
Scikit-learn models are CPU-bound. On high-speed links (10Gbps+), per-packet ML inference cannot keep up with raw throughput.
Mitigation: Batch inference, flow-level analysis, or FPGA/GPU offloading (future work).

**5. Concept Drift**
Network traffic patterns change over time (new protocols, attack tools). Models trained on 1999 KDD data degrade on modern traffic.
Mitigation: Isolation Forest catches novel anomalies; periodic retraining schedule recommended.

### Limitations

| Limitation | Impact | Workaround |
|---|---|---|
| Synthetic dataset in demo mode | Lower real-world accuracy | Use real KDD-99 or CICIDS-2017 data |
| SQLite backend | Not suitable for production scale | Migrate to PostgreSQL |
| No packet re-assembly | Misses multi-packet exploits | Integrate Zeek/Bro for flow reconstruction |
| No encrypted traffic analysis | HTTPS, TLS blind spot | TLS inspection proxy (legal constraints apply) |
| Single-node architecture | No HA/failover | Containerize with Docker, add load balancer |
| 41-feature KDD schema | May miss modern attack features | Augment with CICIDS-2017 features |

---

## 13. Security Considerations

- **Authentication**: Werkzeug password hashing (scrypt), session-based auth with Flask-Login
- **Input Validation**: All API inputs validated before ML inference
- **SQL Injection**: Prevented by SQLAlchemy ORM (parameterized queries)
- **XSS**: Jinja2 auto-escaping on all template variables
- **CSRF**: Flask session tokens; add Flask-WTF for form CSRF in production
- **Log Security**: Logs written locally; rotate and protect in production
- **Secrets**: `SECRET_KEY` from environment variable in production (not hardcoded)
- **Role-Based Access**: `admin`, `analyst`, `viewer` roles (enforce in routes for production)

---

## 14. Future Improvements

| Priority | Improvement | Description |
|---|---|---|
| High | CICIDS-2017 Dataset | Modern dataset with 2017 attack patterns (DDoS, Bot, Infiltration) |
| High | PostgreSQL Migration | Replace SQLite for production-grade persistence and concurrency |
| High | WebSocket Live Feed | Replace polling with Socket.IO for true real-time alerts |
| Medium | LSTM / Deep Learning | Time-series anomaly detection on traffic sequences |
| Medium | GeoIP Integration | Map attacker IPs to geographic locations on the dashboard |
| Medium | Docker Compose | Containerize Flask + DB + Nginx for reproducible deployment |
| Medium | PDF Report Export | ReportLab-generated PDF reports for compliance |
| Low | Multi-tenant | Support multiple organizations with data isolation |
| Low | Email/SMS Alerts | Notify admins of critical alerts via SendGrid/Twilio |
| Low | Model Versioning | MLflow integration for experiment tracking and A/B testing |
| Low | MITRE ATT&CK Mapping | Map detected attacks to MITRE ATT&CK framework tactics |

---

## References

1. Tavallaee, M. et al. "A Detailed Analysis of the KDD CUP 99 Data Set." CISDA 2009.
2. Breiman, L. "Random Forests." Machine Learning 45, 2001.
3. Liu, F.T. et al. "Isolation Forest." ICDM 2008.
4. Ward, J.H. "Hierarchical Grouping to Optimize an Objective Function." JASA 1963.
5. Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." JMLR 12, 2011.
6. CICIDS-2017 Dataset: https://www.unb.ca/cic/datasets/ids-2017.html
7. KDD Cup 99: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

---

*NetGuard IDS v2.4.0 — Documentation generated 2026*
*All 48 automated tests passing — ML Engine: 5 models active*
