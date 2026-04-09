# CreditAI — Enterprise Risk Intelligence Platform
### Full-Stack AI Credit Risk System · Dark Mode · Production Ready

---
Live Demo

Access the platform here

Multi-Agent Architecture

CreditAI leverages a multi-agent style design, where specialized AI modules (“agents”) operate independently but collaborate to deliver comprehensive credit risk assessments. Each agent focuses on a distinct domain:

Agent	Role	Output
Credit ML Agent	Predicts default probability using XGBoost, Random Forest, and Gradient Boosting	Probability score, risk tier
Fraud Detection Agent	Detects anomalies in transaction histories using Isolation Forest	Fraud risk level
NLP Sentiment Agent	Analyzes financial news headlines using TF-IDF + Logistic Regression	Market sentiment, impact score

How it works:

A new credit application or news feed enters the system.
Each agent processes the input in parallel.
Outputs are aggregated using weighted scores (Credit ML 55%, Fraud 25%, NLP 20%) to produce a final AI-driven decision.
The results are displayed on the dashboard and accessible via API endpoints.

Benefits of this design:

Specialization – each agent is optimized for its task.
Parallelism – fast, simultaneous predictions (~200ms per application).
Transparency – outputs from each agent are independently monitorable.
Scalability – new agents (e.g., deep learning modules) can be added seamlessly.

## What's Inside

A complete, end-to-end AI credit risk platform with:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Vanilla HTML/CSS/JS | Dark-mode professional dashboard |
| **Backend** | FastAPI (Python) | REST API, serves frontend |
| **Credit ML** | XGBoost + Random Forest + GBM | Default probability prediction |
| **Fraud Detection** | Isolation Forest | Anomaly detection on transactions |
| **NLP Engine** | TF-IDF + Logistic Regression | Financial news sentiment |
| **Data** | Pandas + synthetic generator | 5,000 applications, 200 sequences |

---

## Project Structure

```
crp-full/
├── backend/
│   ├── main.py                   ← FastAPI app (run this)
│   ├── requirements.txt
│   ├── ml/
│   │   ├── credit_ml_model.py    ← XGBoost + RF + GBM ensemble
│   │   ├── credit_ml_model.pkl   ← trained model (binary)
│   │   ├── nlp_sentiment_engine.py
│   │   ├── nlp_engine.pkl        ← trained NLP model
│   │   ├── fraud_pipeline.pkl    ← trained Isolation Forest
│   │   └── nlp_sentiment_engine.py
│   ├── models/                   ← module alias shims (do not delete)
│   │   ├── __init__.py
│   │   ├── credit_ml_model.py
│   │   └── nlp_sentiment_engine.py
│   └── data/
│       ├── credit_applications.csv   ← 5,000 synthetic applications
│       └── dashboard_metrics.json    ← pre-computed portfolio metrics
│
└── frontend/
    ├── index.html                ← full dark-mode dashboard (source)
    └── dist/
        └── index.html            ← served by FastAPI
```

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Install dependencies

```bash
cd crp-full/backend
pip install -r requirements.txt
```

If you hit permission errors:
```bash
pip install -r requirements.txt --break-system-packages
# or use a virtual environment:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Start the server

```bash
cd crp-full/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
✅ All 3 ML models loaded
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3 — Open in browser

```
http://localhost:8000
```

The dark-mode dashboard opens automatically. The backend serves the frontend.

### Step 4 — API docs (optional)

```
http://localhost:8000/docs
```

Swagger UI — test all endpoints interactively.

---

## Dashboard Features

### Portfolio Dashboard
- Real-time KPI cards (applications, exposure, credit score, default rate)
- Monthly application volume bar chart
- Risk tier donut chart (AAA → B)
- Industry breakdown horizontal bar chart
- Live applications table with decisions

### Model Performance
- AUC comparison for all 4 models
- ROC curve visualization
- Feature importance ranking (10 features)
- Model architecture detail cards

### News Sentiment (NLP)
- Live sentiment analyzer — paste any headlines
- Positive / Neutral / Negative classification
- Credit impact score in basis points
- Per-headline breakdown with confidence scores

### New Application (AI Decision)
- Full credit form with 12 fields
- 3-model AI decision in ~200ms
- Shows: default probability, risk tier, interest rate, approved amount
- Component breakdown: Credit ML 55% + Fraud 25% + Sentiment 20%
- Risk factors + next steps

### Fraud Monitor
- Pipeline visualization (5-step flow)
- Live simulation: Low / Medium / High risk scenarios
- Active alerts panel with probability scores

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + model status |
| GET | `/api/portfolio/summary` | KPI summary |
| GET | `/api/portfolio/risk-tiers` | Tier distribution |
| GET | `/api/portfolio/monthly-volume` | Monthly volume |
| GET | `/api/portfolio/industries` | By industry |
| GET | `/api/portfolio/risk-distribution` | Low/Med/High counts |
| GET | `/api/portfolio/recent-applications` | Recent 10 apps |
| GET | `/api/models/performance` | AUC + feature importance |
| POST | `/api/application/assess` | Full AI credit decision |
| POST | `/api/sentiment/analyze` | NLP headline analysis |
| GET | `/api/fraud/simulate` | Fraud simulation |

---

## Model Performance

| Model | ROC-AUC | Notes |
|-------|---------|-------|
| XGBoost | 0.7758 | 300 trees, depth 6 |
| Random Forest | 0.7767 | 200 trees, balanced |
| Gradient Boosting | 0.7853 | 150 trees, depth 5 |
| **Ensemble** | **0.7820** | Soft voting 50/25/25 |
| NLP (F1) | **1.000** | TF-IDF 3-gram + LogReg |

---

## Extending with HuggingFace / PyTorch / TensorFlow

The models use a clean interface — drop in any replacement:

### FinBERT (HuggingFace) — drop-in for NLP
```python
# In backend/ml/nlp_sentiment_engine.py
from transformers import pipeline as hf_pipeline

class FinBERTEngine:
    def __init__(self):
        self.pipe = hf_pipeline("text-classification",
                                model="ProsusAI/finbert",
                                return_all_scores=True)

    def analyze(self, headlines, company_name=None):
        results = self.pipe(headlines)
        # map to same output format as FinancialNLPEngine
        ...
```

### PyTorch LSTM — drop-in for fraud detection
```python
import torch, torch.nn as nn

class LSTMFraudDetector(nn.Module):
    def __init__(self, input_size=4, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
```

### TensorFlow — deep credit scoring model
```python
import tensorflow as tf

def build_deep_credit_model(n_features=23):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['AUC', 'accuracy'])
    return model
```

---

## Production Deployment

```
                  ┌─────────────┐
    Browser ──────►  nginx      │
                  │  :80/:443   │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  FastAPI    │
                  │  :8000      │
                  │  (uvicorn)  │
                  └──────┬──────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   ┌──────▼──────┐ ┌─────▼─────┐ ┌────▼─────┐
   │ Credit ML   │ │  Fraud    │ │   NLP    │
   │ Ensemble    │ │ Isolat.F. │ │  Engine  │
   └─────────────┘ └───────────┘ └──────────┘

docker-compose up -d
```

## Troubleshooting

**Models not loading:**
```bash
# Make sure you're inside crp-full/backend/
cd crp-full/backend
python -c "import pickle; pickle.load(open('ml/credit_ml_model.pkl','rb'))"
```

**Port already in use:**
```bash
uvicorn main:app --port 8001
# Then open http://localhost:8001
```

**Frontend not loading (API calls failing):**
The frontend auto-detects `localhost` and sends requests to `http://localhost:8000`.
If you run on a different port, update the `API` constant at the top of `frontend/index.html`:
```js
const API = 'http://localhost:YOUR_PORT';
```

**CORS errors in browser:**
The backend already has `allow_origins=["*"]`. If you still see CORS errors,
hard-reload the browser (Ctrl+Shift+R) and ensure the backend is running.

---

## License

MIT — free for commercial and educational use.
