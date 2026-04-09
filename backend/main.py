"""
AI Credit Risk Platform — FastAPI Backend v2.3
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os, sys, json, pickle, warnings, importlib.util
from datetime import datetime
from typing import Optional, List
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE     = os.path.dirname(os.path.abspath(__file__))
ML_DIR   = os.path.join(BASE, "ml")
DATA_DIR = os.path.join(BASE, "data")

# ── Register retrain module so pickled classes resolve ────────────────────────
_retrain_path = os.path.join(BASE, "retrain.py")
_spec = importlib.util.spec_from_file_location("retrain", _retrain_path)
_retrain_mod = importlib.util.module_from_spec(_spec)
for _alias in ["retrain", "__main__", "credit_ml_model", "nlp_sentiment_engine",
               "models.credit_ml_model", "models.nlp_sentiment_engine",
               "ml.credit_ml_model",    "ml.nlp_sentiment_engine"]:
    sys.modules[_alias] = _retrain_mod
_spec.loader.exec_module(_retrain_mod)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="AI Credit Risk Platform", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Load models ───────────────────────────────────────────────────────────────
MODELS: dict = {}

def _load():
    for key, fname in [("credit", "credit_ml_model.pkl"),
                       ("fraud",  "fraud_pipeline.pkl"),
                       ("nlp",    "nlp_engine.pkl")]:
        path = os.path.join(ML_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: Not found: {path}"); continue
        try:
            with open(path, "rb") as f:
                MODELS[key] = pickle.load(f)
        except Exception as e:
            print(f"WARNING: {key} load error: {e}")
    if len(MODELS) == 3:
        print(f"All 3 models loaded: {list(MODELS.keys())}")
    else:
        print(f"Only {len(MODELS)}/3 models loaded. Run: python retrain.py")

_load()

# ── Numpy sanitizer — THE fix for the 500 error ──────────────────────────────
# FastAPI cannot serialize numpy.bool_, numpy.int64, numpy.float32 etc.
# This converts every value coming out of ML models to native Python types.
def _san(obj):
    if isinstance(obj, dict):
        return {k: _san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_san(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_san(v) for v in obj.tolist()]
    return obj

# ── Schemas ───────────────────────────────────────────────────────────────────
class ApplicationRequest(BaseModel):
    company_name: str
    industry: str = "Technology"
    loan_purpose: str = "Business Expansion"
    annual_revenue: float
    loan_amount_requested: float
    years_in_business: float
    num_employees: int = 50
    credit_score: int
    debt_to_income_ratio: float
    monthly_cash_flow: float
    existing_loans: int = 0
    late_payments_12m: int = 0
    avg_daily_transactions: float = 20.0
    transaction_volatility: float = 0.2
    max_overdraft_30d: float = 0.0

class SentimentRequest(BaseModel):
    headlines: List[str]
    company_name: Optional[str] = "Unknown"

# ── Helpers ───────────────────────────────────────────────────────────────────
RATES = {"AAA": 4.5, "AA": 5.0, "A": 5.75, "BBB": 7.0, "BB": 9.5, "B": 13.0}

def _tier(score: int, dti: float) -> str:
    if score >= 750 and dti < 0.30: return "AAA"
    if score >= 700 and dti < 0.45: return "AA"
    if score >= 650 and dti < 0.55: return "A"
    if score >= 600 and dti < 0.65: return "BBB"
    if score >= 550:                 return "BB"
    return "B"

def _txn_seq(base: float, days: int = 90, vol: float = 0.2):
    rng = np.random.default_rng(42)
    return [{"day": d,
             "total_amount":     float(max(0.0, base * rng.normal(1, vol))),
             "num_transactions": int(max(0, rng.poisson(20))),
             "avg_transaction":  float(base / 20),
             "is_weekend":       int(d % 7 >= 5)}
            for d in range(days)]

def _need(key: str):
    if key not in MODELS:
        raise HTTPException(503, f"'{key}' model not loaded. Run: python retrain.py  then restart uvicorn.")

# ── Portfolio endpoints ───────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "models": list(MODELS.keys()), "ts": datetime.utcnow().isoformat()}

@app.get("/api/portfolio/summary")
def summary():
    with open(os.path.join(DATA_DIR, "dashboard_metrics.json")) as f: m = json.load(f)
    p = m["portfolio"]
    return {k: p[k] for k in ["total_applications","total_exposure","avg_credit_score",
                                "default_rate","approved","declined","avg_loan"]}

@app.get("/api/portfolio/risk-tiers")
def risk_tiers():
    with open(os.path.join(DATA_DIR, "dashboard_metrics.json")) as f: m = json.load(f)
    return m["risk_tiers"]

@app.get("/api/portfolio/monthly-volume")
def monthly_vol():
    with open(os.path.join(DATA_DIR, "dashboard_metrics.json")) as f: m = json.load(f)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    mv = m["monthly_applications"]
    return [{"month": months[i], "count": mv.get(f"{i+1:02d}", 0)} for i in range(12)]

@app.get("/api/portfolio/industries")
def industries():
    with open(os.path.join(DATA_DIR, "dashboard_metrics.json")) as f: m = json.load(f)
    return [{"industry": k, "count": v} for k, v in m["industries"].items()]

@app.get("/api/portfolio/risk-distribution")
def risk_dist():
    with open(os.path.join(DATA_DIR, "dashboard_metrics.json")) as f: m = json.load(f)
    return m["risk_distribution"]

@app.get("/api/portfolio/recent-applications")
def recent(limit: int = 10):
    df = pd.read_csv(os.path.join(DATA_DIR, "credit_applications.csv"))
    s = df.sample(min(limit, len(df)), random_state=99)
    rows = []
    for _, r in s.iterrows():
        dp = float(r["default_probability"])
        rows.append({
            "application_id": str(r["application_id"]),
            "company_name":   str(r["company_name"]),
            "industry":       str(r["industry"]),
            "loan_amount":    float(r["loan_amount_requested"]),
            "credit_score":   int(r["credit_score"]),
            "risk_tier":      str(r["risk_tier"]),
            "default_probability": round(dp * 100, 1),
            "decision": "APPROVE" if dp < 0.25 else ("REVIEW" if dp < 0.5 else "DECLINE"),
            "date": str(r["application_date"]),
        })
    return rows

@app.get("/api/models/performance")
def perf():
    return {
        "models": [
            {"name": "XGBoost",           "auc": 0.7758, "color": "#3d82e4"},
            {"name": "Random Forest",     "auc": 0.7767, "color": "#8b7cf6"},
            {"name": "Gradient Boosting", "auc": 0.7853, "color": "#22c55e"},
            {"name": "Ensemble",          "auc": 0.7820, "color": "#f59e0b"},
            {"name": "NLP (F1)",          "auc": 1.000,  "color": "#ef4444"},
        ],
        "ensemble": {"roc_auc": 0.7820, "f1_score": 0.7630, "avg_precision": 0.8516},
        "feature_importance": [
            {"feature": "Late payments (12m)",    "importance": 0.1408},
            {"feature": "Credit score",           "importance": 0.1045},
            {"feature": "Loan-to-revenue ratio",  "importance": 0.0677},
            {"feature": "High DTI x low credit",  "importance": 0.0636},
            {"feature": "Cash flow adequacy",     "importance": 0.0427},
            {"feature": "Debt service coverage",  "importance": 0.0426},
            {"feature": "Debt-to-income ratio",   "importance": 0.0410},
            {"feature": "Experience x credit",    "importance": 0.0376},
            {"feature": "Transaction volatility", "importance": 0.0369},
            {"feature": "Volatility risk",        "importance": 0.0330},
        ],
    }

# ── AI: Credit Decision ───────────────────────────────────────────────────────
@app.post("/api/application/assess")
def assess(req: ApplicationRequest):
    _need("credit")
    _need("fraud")

    app_dict = req.dict()
    app_dict["loan_to_revenue_ratio"] = req.loan_amount_requested / max(req.annual_revenue, 1.0)

    try:
        ml = _san(MODELS["credit"].predict(app_dict))
    except Exception as e:
        raise HTTPException(500, f"Credit ML error: {e}")

    txn = _txn_seq(req.monthly_cash_flow / 30.0, vol=req.transaction_volatility)
    try:
        fraud = _san(MODELS["fraud"].detect(txn))
    except Exception:
        fraud = {"fraud_probability": 0.05, "risk_level": "LOW",
                 "is_fraud": False, "alerts": [], "anomaly_score": 0.0}

    ml_r  = float(ml["default_probability"])
    fr_r  = float(fraud["fraud_probability"])
    comp  = round(min(ml_r * 0.55 + fr_r * 0.25, 1.0), 4)
    tier  = _tier(req.credit_score, req.debt_to_income_ratio)
    base  = RATES[tier]
    fprem = 1.5 if fr_r > 0.5 else (0.5 if fr_r > 0.3 else 0.0)
    rate  = round(base + fprem, 2)

    hd = []
    if fr_r > 0.8:             hd.append("Fraud probability exceeds 80%")
    if ml_r > 0.8:             hd.append("ML default probability exceeds 80%")
    if req.credit_score < 400: hd.append("Credit score below minimum (400)")

    if hd:            st = "DECLINED"
    elif comp < 0.25: st = "APPROVED"
    elif comp < 0.45: st = "CONDITIONAL"
    elif comp < 0.60: st = "UNDER_REVIEW"
    else:             st = "DECLINED"

    amt = (req.loan_amount_requested       if st == "APPROVED"    else
           req.loan_amount_requested * 0.7 if st == "CONDITIONAL" else 0.0)

    steps = {
        "APPROVED":     ["Generate loan agreement documents", "Complete KYC/AML verification", "Disburse funds upon documentation"],
        "CONDITIONAL":  ["Provide additional collateral", "Personal guarantee from directors required", "Monthly cash flow monitoring"],
        "UNDER_REVIEW": ["Schedule credit committee review", "Request 3 years of audited financials", "Obtain independent business valuation"],
        "DECLINED":     ["Issue formal decline letter", "Provide credit improvement recommendations", "Re-application eligible after 6 months"],
    }

    body = {
        "application_id":       f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "company":              req.company_name,
        "timestamp":            datetime.utcnow().isoformat(),
        "status":               st,
        "composite_risk_score": comp,
        "risk_tier":            tier,
        "interest_rate":        rate,
        "base_rate":            base,
        "approved_amount":      round(float(amt), 2),
        "hard_decline_reasons": hd,
        "components": {
            "credit_ml": {
                "default_probability": float(ml["default_probability"]),
                "risk_tier":           str(ml["risk_tier"]),
                "top_risk_factors":    [str(x) for x in ml.get("top_risk_factors", [])],
                "model_predictions":   {k: float(v) for k, v in ml.get("individual_model_predictions", {}).items()},
                "weight": 0.55,
            },
            "fraud_detection": {
                "fraud_probability": float(fraud["fraud_probability"]),
                "risk_level":        str(fraud["risk_level"]),
                "is_suspicious":     bool(fraud.get("is_fraud", False)),
                "alerts":            [str(a) for a in fraud.get("alerts", [])],
                "anomaly_score":     float(fraud.get("anomaly_score", 0.0)),
                "weight": 0.25,
            },
            "news_sentiment": {
                "sentiment_trend":       "STABLE",
                "avg_impact_score":      0.0,
                "credit_adjustment_bps": 0,
                "weight": 0.20,
                "note": "Paste headlines in the News Sentiment tab for live NLP analysis",
            },
        },
        "rate_breakdown":  {"base_rate": base, "fraud_premium": fprem, "final_rate": rate},
        "next_steps":      steps.get(st, []),
    }
    return JSONResponse(content=body)

# ── AI: NLP Sentiment ─────────────────────────────────────────────────────────
@app.post("/api/sentiment/analyze")
def sentiment(req: SentimentRequest):
    _need("nlp")
    if not req.headlines:
        raise HTTPException(400, "Provide at least one headline")
    try:
        return JSONResponse(content=_san(MODELS["nlp"].analyze(req.headlines, req.company_name)))
    except Exception as e:
        raise HTTPException(500, str(e))

# ── AI: Fraud Simulation ──────────────────────────────────────────────────────
@app.get("/api/fraud/simulate")
def fraud_sim(company_id: str = "COMP-0001", risk_level: str = "low"):
    _need("fraud")

    seed = abs(hash(company_id)) % (2 ** 31)
    rng  = np.random.default_rng(seed)
    base = float(rng.lognormal(7, 1))

    if risk_level == "high":
        amounts = [base * float(rng.normal(1, 0.08)) for _ in range(83)] + [base * 0.02] * 7
    elif risk_level == "medium":
        amounts = [base * float(rng.normal(1, 0.25)) for _ in range(90)]
        amounts[44] = base * 9.0
        amounts[45] = base * 7.5
    else:
        amounts = [base * float(rng.normal(1, 0.12)) for _ in range(90)]

    seq = [{"day": d,
            "total_amount":     float(max(0.0, a)),
            "num_transactions": int(max(0, int(rng.poisson(15)))),
            "avg_transaction":  float(max(0.0, a)) / 15.0,
            "is_weekend":       int(d % 7 >= 5)}
           for d, a in enumerate(amounts)]

    try:
        det = _san(MODELS["fraud"].detect(seq))
        body = {
            "company_id":           company_id,
            "risk_level_simulated": risk_level,
            "sequence_days":        len(seq),
            "detection": {
                "is_fraud":          bool(det.get("is_fraud", False)),
                "fraud_probability": float(det.get("fraud_probability", 0.0)),
                "anomaly_score":     float(det.get("anomaly_score", 0.0)),
                "risk_level":        str(det.get("risk_level", "LOW")),
                "alerts":            [str(a) for a in det.get("alerts", [])],
                "requires_review":   bool(det.get("requires_review", False)),
            },
            "sample_amounts": [round(float(s["total_amount"]), 2) for s in seq[:14]],
        }
        return JSONResponse(content=body)
    except Exception as e:
        raise HTTPException(500, f"Fraud detection error: {e}")

# ── Serve frontend ────────────────────────────────────────────────────────────
_dist = os.path.join(BASE, "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
else:
    @app.get("/")
    def root():
        return {"message": "CreditAI API v2.3 — /docs for Swagger UI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
