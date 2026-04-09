"""
CreditAI Platform — Retrain All Models
Run this ONCE on your machine to regenerate compatible .pkl files.

Usage:
    cd crp-full/backend
    python retrain.py
"""

import os, sys, json, pickle, random, warnings, re
warnings.filterwarnings("ignore")

# ── Register this module as 'retrain' immediately so pickle uses that name.
# This must happen before any class definitions.
import importlib.util as _ilu
if __name__ == '__main__':
    _self_path = os.path.abspath(__file__)
    _spec = _ilu.spec_from_file_location("retrain", _self_path)
    _mod  = _ilu.module_from_spec(_spec)
    sys.modules.setdefault("retrain", _mod)
    # Populate __module__ for classes defined below
    _MODNAME = "retrain"
else:
    _MODNAME = __name__

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, IsolationForest)
from sklearn.preprocessing import RobustScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

BASE = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE, "ml")
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

fake = Faker()
np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Generate synthetic data
# ═══════════════════════════════════════════════════════════════

def generate_credit_applications(n=5000):
    print(f"  Generating {n} credit applications...")
    industries = ['Technology','Healthcare','Finance','Retail','Manufacturing',
                  'Real Estate','Energy','Transportation','Agriculture','Media']
    loan_purposes = ['Business Expansion','Working Capital','Equipment Purchase',
                     'Real Estate','Debt Consolidation','Research & Development']
    records = []
    for i in range(n):
        annual_revenue   = np.random.lognormal(mean=13, sigma=2)
        years_in_biz     = max(0, np.random.normal(8, 5))
        credit_score     = int(np.clip(np.random.normal(680, 80), 300, 850))
        dti              = float(np.clip(np.random.beta(2, 5), 0.01, 0.95))
        loan_amount      = np.random.lognormal(mean=11, sigma=1.5)
        num_employees    = max(1, int(np.random.lognormal(3, 1.5)))
        ltr_ratio        = loan_amount / max(annual_revenue, 1)
        monthly_cf       = annual_revenue / 12 * np.random.uniform(0.05, 0.25)
        existing_loans   = np.random.poisson(1.5)
        late_payments    = np.random.poisson(0.8)
        industry         = random.choice(industries)
        avg_txns         = np.random.lognormal(3, 1)
        txn_vol          = np.random.uniform(0.1, 0.8)
        max_od           = np.random.exponential(500) if np.random.random() < 0.3 else 0.0

        risk = 0
        risk += max(0, (700 - credit_score) / 100) * 2
        risk += dti * 3
        risk += min(ltr_ratio, 5) * 0.5
        risk += late_payments * 0.8
        risk += (1 / max(years_in_biz, 0.5)) * 0.5
        risk += txn_vol * 1.5
        risk += (1 if max_od > 1000 else 0) * 0.5
        default_prob = 1 / (1 + np.exp(-risk + 3))
        default = 1 if np.random.random() < default_prob else 0

        if credit_score >= 750 and dti < 0.3:   tier = 'AAA'
        elif credit_score >= 700 and dti < 0.45: tier = 'AA'
        elif credit_score >= 650 and dti < 0.55: tier = 'A'
        elif credit_score >= 600 and dti < 0.65: tier = 'BBB'
        elif credit_score >= 550:                tier = 'BB'
        else:                                    tier = 'B'

        records.append({
            'application_id':       f'APP-{i+1:06d}',
            'company_name':         fake.company(),
            'industry':             industry,
            'loan_purpose':         random.choice(loan_purposes),
            'annual_revenue':       round(annual_revenue, 2),
            'loan_amount_requested': round(loan_amount, 2),
            'loan_to_revenue_ratio': round(ltr_ratio, 4),
            'years_in_business':    round(years_in_biz, 1),
            'num_employees':        num_employees,
            'credit_score':         credit_score,
            'debt_to_income_ratio': round(dti, 4),
            'monthly_cash_flow':    round(monthly_cf, 2),
            'existing_loans':       existing_loans,
            'late_payments_12m':    late_payments,
            'avg_daily_transactions': round(avg_txns, 2),
            'transaction_volatility': round(txn_vol, 4),
            'max_overdraft_30d':    round(max_od, 2),
            'risk_tier':            tier,
            'default_probability':  round(default_prob, 4),
            'default':              default,
            'application_date': (datetime.now() - timedelta(days=random.randint(0,365))).strftime('%Y-%m-%d'),
        })
    df = pd.DataFrame(records)
    print(f"  ✓ {len(df)} apps | default rate: {df['default'].mean():.1%}")
    return df


def generate_transaction_sequences(n=200, seq_len=90):
    print(f"  Generating {n} transaction sequences ({seq_len} days each)...")
    seqs = []
    for i in range(n):
        base = np.random.lognormal(7, 1.5)
        trend = np.random.uniform(-0.002, 0.003)
        amp = np.random.uniform(0, 0.3)
        seq = []
        for t in range(seq_len):
            dow = t % 7
            wf = 0.3 if dow >= 5 else 1.0
            seasonal = 1 + amp * np.sin(2 * np.pi * t / 30)
            amount = base * wf * seasonal * (1 + trend * t) * np.random.normal(1, 0.15)
            seq.append({
                'day': t,
                'total_amount':    round(float(max(0, amount)), 2),
                'num_transactions': max(0, int(np.random.poisson(15) * wf)),
                'avg_transaction':  round(float(max(0, amount)) / 15, 2),
                'is_weekend':       int(dow >= 5),
            })
        label = 1 if (trend < -0.001 and base < 1000) else 0
        seqs.append({'company_id': f'COMP-{i:04d}', 'sequence': seq, 'fraud_label': label})
    print(f"  ✓ {len(seqs)} sequences")
    return seqs


def generate_news(n=500):
    print(f"  Generating {n} financial news headlines...")
    pos = ["{c} reports record quarterly earnings, beating analyst expectations by {p}%",
           "{c} secures ${a}M funding round, expanding to new markets",
           "{c} credit rating upgraded to {r} by major agencies",
           "{c} reports strong cash flow growth of {p}% year-over-year",
           "{c} announces strategic partnership, stock surges {p}%"]
    neg = ["{c} misses earnings targets, stock falls {p}% in after-hours trading",
           "{c} faces liquidity concerns as debt levels rise {p}%",
           "{c} credit rating downgraded amid restructuring concerns",
           "{c} reports {p}% decline in revenue, layoffs announced",
           "{c} under regulatory investigation for financial irregularities"]
    neu = ["{c} releases annual report, revenue in line with expectations",
           "{c} appoints new CFO as part of leadership transition",
           "{c} announces quarterly dividend of ${a} per share",
           "{c} to present at upcoming investor conference"]
    rows = []
    for i in range(n):
        c = fake.company(); p = round(random.uniform(2,35),1)
        a = random.randint(10,500); r = random.choice(['AA+','AA','A+','A','BBB+'])
        roll = random.random()
        if roll < 0.4:
            h = random.choice(pos); s = 'positive'; sc = round(random.uniform(0.6,1.0),3)
        elif roll < 0.7:
            h = random.choice(neg); s = 'negative'; sc = round(random.uniform(0.0,0.4),3)
        else:
            h = random.choice(neu); s = 'neutral';  sc = round(random.uniform(0.35,0.65),3)
        rows.append({
            'news_id': f'NEWS-{i+1:05d}',
            'headline': h.format(c=c,p=p,a=a,r=r),
            'company': c, 'sentiment_label': s, 'sentiment_score': sc,
        })
    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} headlines")
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Credit Risk ML Pipeline
# ═══════════════════════════════════════════════════════════════

class CreditRiskMLPipeline:
    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.models = {}
        self.ensemble = None
        self.feature_names = None
        self.feature_importance_scores = {}
        self.metrics = {}
        self.is_trained = False

    def _engineer(self, df):
        df = df.copy()
        df['debt_service_coverage']       = df['monthly_cash_flow'] / (df['loan_amount_requested'] / 60 + 1)
        df['revenue_per_employee']        = df['annual_revenue'] / df['num_employees'].clip(lower=1)
        df['loan_per_employee']           = df['loan_amount_requested'] / df['num_employees'].clip(lower=1)
        df['credit_utilization_proxy']    = df['existing_loans'] / (df['credit_score'] / 100 + 1)
        df['high_dti_low_credit']         = (df['debt_to_income_ratio'] * (800 - df['credit_score'])) / 100
        df['cash_flow_adequacy']          = df['monthly_cash_flow'] / (df['loan_amount_requested'] / 36 + 1)
        df['experience_credit_interaction']= df['years_in_business'] * (df['credit_score'] / 700)
        df['volatility_risk']             = df['transaction_volatility'] * df['max_overdraft_30d'].clip(upper=5000) / 1000
        for col in ['industry','loan_purpose']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        return df

    def _features(self):
        return ['annual_revenue','loan_amount_requested','loan_to_revenue_ratio',
                'years_in_business','num_employees','credit_score','debt_to_income_ratio',
                'monthly_cash_flow','existing_loans','late_payments_12m',
                'avg_daily_transactions','transaction_volatility','max_overdraft_30d',
                'debt_service_coverage','revenue_per_employee','loan_per_employee',
                'credit_utilization_proxy','high_dti_low_credit','cash_flow_adequacy',
                'experience_credit_interaction','volatility_risk',
                'industry_encoded','loan_purpose_encoded']

    def train(self, df):
        print("\n  [Credit ML] Engineering features...")
        df_e = self._engineer(df)
        self.feature_names = self._features()
        X = df_e[self.feature_names].fillna(0)
        y = df_e['default']

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_te_s = self.scaler.transform(X_te)

        spw = (y_tr == 0).sum() / (y_tr == 1).sum()

        print("  [Credit ML] Training XGBoost...")
        xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    scale_pos_weight=spw, eval_metric='auc',
                                    random_state=42, verbosity=0)
        xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
        auc_xgb = roc_auc_score(y_te, xgb_m.predict_proba(X_te_s)[:,1])
        print(f"      XGBoost  AUC: {auc_xgb:.4f}")

        print("  [Credit ML] Training Random Forest...")
        rf_m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=42, n_jobs=-1)
        rf_m.fit(X_tr_s, y_tr)
        auc_rf = roc_auc_score(y_te, rf_m.predict_proba(X_te_s)[:,1])
        print(f"      RF       AUC: {auc_rf:.4f}")

        print("  [Credit ML] Training Gradient Boosting...")
        gb_m = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08,
                                           subsample=0.8, min_samples_leaf=10, random_state=42)
        gb_m.fit(X_tr_s, y_tr)
        auc_gb = roc_auc_score(y_te, gb_m.predict_proba(X_te_s)[:,1])
        print(f"      GB       AUC: {auc_gb:.4f}")

        print("  [Credit ML] Building ensemble...")
        self.ensemble = VotingClassifier(
            estimators=[('xgb',xgb_m),('rf',rf_m),('gb',gb_m)],
            voting='soft', weights=[0.5,0.25,0.25]
        )
        self.ensemble.fit(X_tr_s, y_tr)
        self.models = {'xgboost': xgb_m, 'random_forest': rf_m, 'gradient_boosting': gb_m}

        y_p = self.ensemble.predict_proba(X_te_s)[:,1]
        y_c = (y_p >= 0.5).astype(int)
        ens_auc = roc_auc_score(y_te, y_p)
        print(f"      Ensemble AUC: {ens_auc:.4f}  F1: {f1_score(y_te,y_c):.4f}")

        fi = xgb_m.feature_importances_
        self.feature_importance_scores = dict(sorted(
            zip(self.feature_names, fi), key=lambda x: x[1], reverse=True))
        self.is_trained = True
        self.metrics = {'ensemble_auc': round(ens_auc,4), 'xgb_auc': round(auc_xgb,4),
                        'rf_auc': round(auc_rf,4), 'gb_auc': round(auc_gb,4)}
        return self.metrics

    def predict(self, app_dict):
        df = pd.DataFrame([app_dict])
        df_e = self._engineer(df)
        X = df_e[self.feature_names].fillna(0)
        X_s = self.scaler.transform(X)
        proba = self.ensemble.predict_proba(X_s)[0]
        dp = float(proba[1])
        ind = {n: round(float(m.predict_proba(X_s)[0][1]),4) for n,m in self.models.items()}

        if   dp < 0.15: tier,rec,prem = 'AAA','APPROVE',0.5
        elif dp < 0.25: tier,rec,prem = 'AA', 'APPROVE',1.0
        elif dp < 0.35: tier,rec,prem = 'A',  'APPROVE',1.5
        elif dp < 0.50: tier,rec,prem = 'BBB','REVIEW', 2.5
        elif dp < 0.65: tier,rec,prem = 'BB', 'DECLINE',4.0
        else:           tier,rec,prem = 'B',  'DECLINE',6.0

        flags = []
        if app_dict.get('credit_score',700) < 600:         flags.append('Low credit score')
        if app_dict.get('debt_to_income_ratio',0) > 0.5:   flags.append('High debt-to-income ratio')
        if app_dict.get('late_payments_12m',0) > 2:        flags.append('Multiple late payments')
        if app_dict.get('years_in_business',5) < 2:        flags.append('Limited business history')
        if app_dict.get('transaction_volatility',0) > 0.6: flags.append('High transaction volatility')

        return {
            'default_probability': round(dp,4), 'risk_tier': tier,
            'recommendation': rec, 'interest_rate_premium_bps': prem*100,
            'individual_model_predictions': ind, 'top_risk_factors': flags,
            'confidence': round(abs(dp-0.5)*2, 4),
        }


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Fraud Detection
# ═══════════════════════════════════════════════════════════════

class FraudDetectionPipeline:
    def __init__(self):
        self.iso = IsolationForest(contamination=0.05, n_estimators=200, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.thresholds = {}
        self.is_trained = False

    def _extract(self, seqs):
        rows = []
        for sd in seqs:
            seq = sd['sequence']
            amounts = [d['total_amount'] for d in seq]
            txns    = [d['num_transactions'] for d in seq]
            rows.append({
                'mean_amount':   np.mean(amounts),
                'std_amount':    np.std(amounts),
                'cv_amount':     np.std(amounts)/(np.mean(amounts)+1),
                'max_amount':    np.max(amounts),
                'min_amount':    np.min(amounts),
                'trend':         np.polyfit(range(len(amounts)), amounts, 1)[0],
                'mean_txns':     np.mean(txns),
                'std_txns':      np.std(txns),
                'zero_days':     sum(1 for a in amounts if a < 10),
                'spike_days':    sum(1 for a in amounts if a > np.mean(amounts)+2*np.std(amounts)),
                'last7_vs_all':  np.mean(amounts[-7:])/(np.mean(amounts)+1),
                'weekend_ratio': np.mean([d['is_weekend'] for d in seq]),
            })
        return pd.DataFrame(rows)

    def train(self, seqs):
        print("\n  [Fraud] Training Isolation Forest...")
        F = self._extract(seqs)
        X = self.scaler.fit_transform(F.fillna(0))
        self.iso.fit(X)
        scores = self.iso.score_samples(X)
        self.thresholds = {
            'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
            'p5':   float(np.percentile(scores,5)), 'p1': float(np.percentile(scores,1)),
        }
        self.is_trained = True
        labeled = sum(s['fraud_label'] for s in seqs)
        print(f"  ✓ {len(seqs)} sequences | {labeled} known fraud cases")

    def detect(self, seq):
        F = self._extract([{'sequence': seq}])
        X = self.scaler.transform(F.fillna(0))
        score = float(self.iso.score_samples(X)[0])
        pred  = self.iso.predict(X)[0]
        rng   = self.thresholds['mean'] - self.thresholds['p1']
        ns    = max(0, min(100, (self.thresholds['mean']-score)/(rng+0.001)*100))

        alerts = []
        amounts = [d['total_amount'] for d in seq]
        if len(amounts) >= 7:
            if np.mean(amounts[-7:]) < 0.3*np.mean(amounts[:-7]):
                alerts.append('Sudden revenue drop (>70%) in last 7 days')
            if max(amounts[-7:]) > 5*np.mean(amounts[:-7]):
                alerts.append('Suspicious transaction spike detected')
        zd = sum(1 for a in amounts if a < 10)
        if zd > len(amounts)*0.3:
            alerts.append(f'Excessive zero-activity days: {zd}')

        return {
            'is_fraud':          pred == -1 or ns > 75,
            'fraud_probability': round(ns/100, 4),
            'anomaly_score':     round(score, 4),
            'risk_level':        'HIGH' if ns>75 else ('MEDIUM' if ns>40 else 'LOW'),
            'alerts':            alerts,
            'requires_review':   ns>50 or len(alerts)>0,
        }


# ═══════════════════════════════════════════════════════════════
# STEP 4 — NLP Sentiment Engine
# ═══════════════════════════════════════════════════════════════

POSITIVE_TERMS = {
    'record':2.0,'beat':1.8,'exceeded':1.8,'surpassed':1.8,'upgraded':2.0,
    'growth':1.5,'expansion':1.5,'profit':1.5,'strong':1.5,'robust':1.5,
    'raised guidance':2.0,'dividend':1.3,'buyback':1.3,'outperform':1.8,
    'overweight':1.5,'buy rating':2.0,'secured funding':2.0,'approved':1.5,
}
NEGATIVE_TERMS = {
    'miss':-2.0,'missed':-2.0,'decline':-1.8,'fell':-1.5,'downgraded':-2.0,
    'loss':-2.0,'debt':-1.3,'default':-2.5,'bankruptcy':-3.0,'liquidation':-3.0,
    'restructuring':-1.8,'investigation':-2.0,'fraud':-2.5,'layoffs':-1.8,
    'cuts':-1.5,'liquidity':-1.5,'cash burn':-2.0,'insolvency':-2.8,
}

class FinancialNLPEngine:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.metrics = {}

    def _preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r'\$[\d,.]+[mbk]?','[AMOUNT]',text)
        text = re.sub(r'[\d,.]+%','[PERCENT]',text)
        text = re.sub(r'[^\w\s\-]',' ',text)
        return re.sub(r'\s+',' ',text).strip()

    def _lexicon_score(self, text):
        t = text.lower(); score = 0.0
        for term,w in POSITIVE_TERMS.items():
            if term in t: score += w
        for term,w in NEGATIVE_TERMS.items():
            if term in t: score += w
        return float(np.clip(score/5, -1, 1))

    def train(self, news_df):
        print("\n  [NLP] Training sentiment engine...")
        texts  = news_df['headline'].apply(self._preprocess).tolist()
        labels = self.label_encoder.fit_transform(news_df['sentiment_label'].tolist())

        X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,3), max_features=15000,
                                      min_df=2, sublinear_tf=True,
                                      token_pattern=r'\b[a-zA-Z][a-zA-Z\-]+\b')),
            ('clf',   LogisticRegression(C=1.0, class_weight='balanced',
                                          max_iter=500, random_state=42, solver='lbfgs')),
        ])
        cv = cross_val_score(self.pipeline, X_tr, y_tr, cv=5, scoring='f1_macro')
        print(f"      5-fold CV F1: {cv.mean():.4f} ± {cv.std():.4f}")
        self.pipeline.fit(X_tr, y_tr)
        y_pred = self.pipeline.predict(X_te)
        acc = (y_pred == y_te).mean()
        f1m = f1_score(y_te, y_pred, average='macro')
        print(f"      Accuracy: {acc:.4f}  F1-macro: {f1m:.4f}")
        self.metrics = {'accuracy': round(acc,4), 'f1_macro': round(f1m,4)}
        self.is_trained = True
        return self.metrics

    def analyze(self, headlines, company_name=None):
        processed = [self._preprocess(h) for h in headlines]
        probas = self.pipeline.predict_proba(processed)
        preds  = self.pipeline.predict(processed)
        results = []
        for h, pred, proba in zip(headlines, preds, probas):
            label = self.label_encoder.inverse_transform([pred])[0]
            lex   = self._lexicon_score(h)
            neg_p = float(proba[self.label_encoder.transform(['negative'])[0]])
            pos_p = float(proba[self.label_encoder.transform(['positive'])[0]])
            impact = (pos_p - neg_p)*0.7 + lex*0.3
            results.append({
                'headline': h, 'sentiment': label,
                'confidence': round(float(max(proba)),4),
                'probabilities': {cls: round(float(p),4)
                                  for cls,p in zip(self.label_encoder.classes_, proba)},
                'lexicon_score':       round(lex,4),
                'credit_impact_score': round(impact,4),
                'credit_signal': 'POSITIVE' if impact>0.2 else ('NEGATIVE' if impact<-0.2 else 'NEUTRAL'),
            })
        scores = [r['credit_impact_score'] for r in results]
        pos_c  = sum(1 for r in results if r['sentiment']=='positive')
        neg_c  = sum(1 for r in results if r['sentiment']=='negative')
        avg    = float(np.mean(scores))
        return {
            'company':           company_name or 'Unknown',
            'total_headlines':   len(results),
            'positive_count':    pos_c,
            'negative_count':    neg_c,
            'neutral_count':     len(results)-pos_c-neg_c,
            'avg_credit_impact': round(avg,4),
            'sentiment_trend':   'BULLISH' if avg>0.1 else ('BEARISH' if avg<-0.1 else 'STABLE'),
            'credit_adjustment_bps': round(avg*-100, 1),
            'individual_results': results,
        }


class NewsAggregator:
    def __init__(self, nlp): self.nlp = nlp
    def get_credit_sentiment_report(self, company, headlines):
        if not headlines:
            return {'company':company,'status':'NO_DATA','credit_adjustment_bps':0}
        a = self.nlp.analyze(headlines, company)
        flags = []
        if a['negative_count']/max(len(headlines),1) > 0.6: flags.append('HIGH_NEGATIVE_NEWS_RATIO')
        if a['avg_credit_impact'] < -0.4:                   flags.append('SEVERE_NEGATIVE_SENTIMENT')
        if a['sentiment_trend'] == 'BEARISH':               flags.append('BEARISH_MARKET_SENTIMENT')
        rec = ('Manual review required' if 'SEVERE_NEGATIVE_SENTIMENT' in flags
               else 'Increase rate premium by 50-100bps' if 'HIGH_NEGATIVE_NEWS_RATIO' in flags
               else 'Favorable news sentiment' if a['sentiment_trend']=='BULLISH'
               else 'Neutral — proceed with standard assessment')
        return {'company':company,'sentiment_summary':a['sentiment_trend'],
                'avg_impact_score':a['avg_credit_impact'],
                'credit_adjustment_bps':a['credit_adjustment_bps'],
                'risk_flags':flags,'headlines_analyzed':len(headlines),
                'breakdown':{'positive':a['positive_count'],'negative':a['negative_count'],
                             'neutral':a['neutral_count']},
                'recommendation':rec}


# ═══════════════════════════════════════════════════════════════
# MAIN — Run everything
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  CreditAI Platform — Model Retrainer")
    print(f"  Python {sys.version.split()[0]} | numpy {np.__version__}")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/4] Generating synthetic data...")
    df   = generate_credit_applications(5000)
    seqs = generate_transaction_sequences(200, 90)
    news = generate_news(500)

    csv_path = os.path.join(DATA_DIR, "credit_applications.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    seq_path = os.path.join(DATA_DIR, "transaction_sequences.json")
    with open(seq_path, 'w') as f: json.dump(seqs, f)
    print(f"  Saved: {seq_path}")

    # 2. Train Credit ML
    print("\n[2/4] Training Credit Risk ML Pipeline...")
    credit_model = CreditRiskMLPipeline()
    credit_model.train(df)
    cm_path = os.path.join(ML_DIR, "credit_ml_model.pkl")
    with open(cm_path, 'wb') as f: pickle.dump(credit_model, f, protocol=4)
    print(f"  Saved: {cm_path}")

    # 3. Train Fraud Detection
    print("\n[3/4] Training Fraud Detection Pipeline...")
    fraud_model = FraudDetectionPipeline()
    fraud_model.train(seqs)
    fp_path = os.path.join(ML_DIR, "fraud_pipeline.pkl")
    with open(fp_path, 'wb') as f: pickle.dump(fraud_model, f, protocol=4)
    print(f"  Saved: {fp_path}")

    # 4. Train NLP
    print("\n[4/4] Training NLP Sentiment Engine...")
    nlp_model = FinancialNLPEngine()
    nlp_model.train(news)
    nlp_path = os.path.join(ML_DIR, "nlp_engine.pkl")
    with open(nlp_path, 'wb') as f: pickle.dump(nlp_model, f, protocol=4)
    print(f"  Saved: {nlp_path}")

    # 5. Save dashboard metrics
    print("\n[5/5] Computing dashboard metrics...")
    risk_dist = {
        'low':    int((df['default_probability'] < 0.25).sum()),
        'medium': int(((df['default_probability'] >= 0.25) & (df['default_probability'] < 0.5)).sum()),
        'high':   int((df['default_probability'] >= 0.5).sum()),
    }
    monthly = {str(m): int(v) for m,v in
               df.groupby(df['application_date'].str[5:7])['application_id'].count().items()}
    metrics = {
        'portfolio': {
            'total_applications': int(len(df)), 'approved': int((df['default']==0).sum()),
            'declined': int((df['default']==1).sum()), 'default_rate': float(round(df['default'].mean(),4)),
            'avg_loan': float(round(df['loan_amount_requested'].mean(),0)),
            'total_exposure': float(round(df['loan_amount_requested'].sum(),0)),
            'avg_credit_score': float(round(df['credit_score'].mean(),1)),
        },
        'risk_tiers':          {k: int(v) for k,v in df['risk_tier'].value_counts().items()},
        'industries':          {k: int(v) for k,v in df['industry'].value_counts().head(8).items()},
        'model_performance':   credit_model.metrics,
        'feature_importance':  {k: float(v) for k,v in list(credit_model.feature_importance_scores.items())[:10]},
        'monthly_applications': monthly,
        'risk_distribution':   risk_dist,
    }
    dm_path = os.path.join(DATA_DIR, "dashboard_metrics.json")
    with open(dm_path, 'w') as f: json.dump(metrics, f, indent=2)
    print(f"  Saved: {dm_path}")

    # 6. Quick smoke test
    print("\n[Smoke Test] Running quick prediction checks...")
    sample = {
        'company_name':'TestCo','industry':'Technology','loan_purpose':'Business Expansion',
        'annual_revenue':3e6,'loan_amount_requested':500000,'loan_to_revenue_ratio':0.167,
        'years_in_business':6,'num_employees':60,'credit_score':720,
        'debt_to_income_ratio':0.30,'monthly_cash_flow':50000,'existing_loans':1,
        'late_payments_12m':0,'avg_daily_transactions':20,'transaction_volatility':0.2,
        'max_overdraft_30d':0,
    }
    r  = credit_model.predict(sample)
    txn = [{'day':d,'total_amount':1500.0,'num_transactions':20,
            'avg_transaction':75,'is_weekend':int(d%7>=5)} for d in range(90)]
    fr = fraud_model.detect(txn)
    nl = nlp_model.analyze(['Company reports record earnings beating expectations'], 'TestCo')

    print(f"  Credit: tier={r['risk_tier']}  prob={r['default_probability']:.1%}")
    print(f"  Fraud:  level={fr['risk_level']}  prob={fr['fraud_probability']:.1%}")
    print(f"  NLP:    trend={nl['sentiment_trend']}  impact={nl['avg_credit_impact']:.3f}")

    print()
    print("=" * 60)
    print("  ✅ ALL MODELS RETRAINED SUCCESSFULLY!")
    print("  Now run: uvicorn main:app --reload --port 8000")
    print("=" * 60)
