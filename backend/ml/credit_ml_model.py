"""
Enterprise Credit Risk Platform - Module 1: Traditional ML Models
XGBoost + Random Forest Ensemble with full ML pipeline
Scikit-learn based with SHAP explainability
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             precision_recall_curve, average_precision_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
import xgboost as xgb
import pickle
import os

class CreditRiskMLPipeline:
    """
    Enterprise ML Pipeline for Credit Risk Assessment
    Combines XGBoost + Random Forest + Gradient Boosting in a voting ensemble
    """

    def __init__(self):
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.models = {}
        self.ensemble = None
        self.feature_names = None
        self.feature_importance_scores = {}
        self.metrics = {}
        self.is_trained = False

    def _engineer_features(self, df):
        """Advanced feature engineering"""
        df = df.copy()
        
        # Financial ratios
        df['debt_service_coverage'] = df['monthly_cash_flow'] / (df['loan_amount_requested'] / 60 + 1)
        df['revenue_per_employee'] = df['annual_revenue'] / df['num_employees'].clip(lower=1)
        df['loan_per_employee'] = df['loan_amount_requested'] / df['num_employees'].clip(lower=1)
        df['credit_utilization_proxy'] = df['existing_loans'] / (df['credit_score'] / 100 + 1)
        
        # Risk interaction features
        df['high_dti_low_credit'] = (df['debt_to_income_ratio'] * (800 - df['credit_score'])) / 100
        df['cash_flow_adequacy'] = df['monthly_cash_flow'] / (df['loan_amount_requested'] / 36 + 1)
        df['experience_credit_interaction'] = df['years_in_business'] * (df['credit_score'] / 700)
        df['volatility_risk'] = df['transaction_volatility'] * df['max_overdraft_30d'].clip(upper=5000) / 1000
        
        # Categorical encoding
        for col in ['industry', 'loan_purpose']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df

    def _get_feature_columns(self):
        return [
            'annual_revenue', 'loan_amount_requested', 'loan_to_revenue_ratio',
            'years_in_business', 'num_employees', 'credit_score',
            'debt_to_income_ratio', 'monthly_cash_flow', 'existing_loans',
            'late_payments_12m', 'avg_daily_transactions', 'transaction_volatility',
            'max_overdraft_30d', 'debt_service_coverage', 'revenue_per_employee',
            'loan_per_employee', 'credit_utilization_proxy', 'high_dti_low_credit',
            'cash_flow_adequacy', 'experience_credit_interaction', 'volatility_risk',
            'industry_encoded', 'loan_purpose_encoded'
        ]

    def train(self, df):
        print("\n" + "="*60)
        print("  TRAINING TRADITIONAL ML MODELS")
        print("="*60)
        
        # Feature engineering
        print("\n[1/5] Engineering features...")
        df_eng = self._engineer_features(df)
        self.feature_names = self._get_feature_columns()
        
        X = df_eng[self.feature_names].fillna(0)
        y = df_eng['default']
        
        print(f"      Features: {len(self.feature_names)} | Samples: {len(X)}")
        print(f"      Class balance: {y.mean():.1%} default rate")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # ── Model 1: XGBoost ──────────────────────────────────────────
        print("\n[2/5] Training XGBoost...")
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc', random_state=42,
            use_label_encoder=False, verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train,
                      eval_set=[(X_test_scaled, y_test)],
                      verbose=False)
        xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])
        print(f"      XGBoost AUC: {xgb_auc:.4f}")
        self.models['xgboost'] = xgb_model

        # ── Model 2: Random Forest ────────────────────────────────────
        print("\n[3/5] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1])
        print(f"      Random Forest AUC: {rf_auc:.4f}")
        self.models['random_forest'] = rf_model

        # ── Model 3: Gradient Boosting ────────────────────────────────
        print("\n[4/5] Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test_scaled)[:, 1])
        print(f"      Gradient Boosting AUC: {gb_auc:.4f}")
        self.models['gradient_boosting'] = gb_model

        # ── Soft Voting Ensemble ──────────────────────────────────────
        print("\n[5/5] Building ensemble & evaluating...")
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',
            weights=[0.5, 0.25, 0.25]
        )
        self.ensemble.fit(X_train_scaled, y_train)
        
        # Final evaluation
        y_pred_proba = self.ensemble.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        ensemble_auc = roc_auc_score(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Feature importance
        fi = xgb_model.feature_importances_
        self.feature_importance_scores = dict(sorted(
            zip(self.feature_names, fi), key=lambda x: x[1], reverse=True
        ))
        
        self.metrics = {
            'ensemble_auc': round(ensemble_auc, 4),
            'average_precision': round(ap_score, 4),
            'f1_score': round(f1, 4),
            'xgb_auc': round(xgb_auc, 4),
            'rf_auc': round(rf_auc, 4),
            'gb_auc': round(gb_auc, 4),
            'test_samples': len(X_test),
            'train_samples': len(X_train),
            'top_features': list(self.feature_importance_scores.keys())[:10]
        }
        
        self.is_trained = True
        self._print_results(y_test, y_pred, y_pred_proba)
        return self.metrics

    def _print_results(self, y_test, y_pred, y_pred_proba):
        print("\n" + "─"*50)
        print("  ENSEMBLE MODEL RESULTS")
        print("─"*50)
        print(f"  ROC-AUC Score:        {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"  Avg Precision Score:  {average_precision_score(y_test, y_pred_proba):.4f}")
        print(f"  F1 Score:             {f1_score(y_test, y_pred):.4f}")
        print("\n  Top 5 Risk Factors:")
        for i, (feat, imp) in enumerate(list(self.feature_importance_scores.items())[:5], 1):
            bar = "█" * int(imp * 100)
            print(f"  {i}. {feat:<35} {imp:.4f} {bar}")
        print("─"*50)

    def predict(self, application_dict):
        """Predict default risk for a single application"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        df = pd.DataFrame([application_dict])
        df_eng = self._engineer_features(df)
        X = df_eng[self.feature_names].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        proba = self.ensemble.predict_proba(X_scaled)[0]
        default_prob = proba[1]
        
        # Individual model predictions
        individual_preds = {}
        for name, model in self.models.items():
            p = model.predict_proba(X_scaled)[0][1]
            individual_preds[name] = round(float(p), 4)
        
        # Risk tier assignment
        if default_prob < 0.15:
            risk_tier = 'AAA'
            recommendation = 'APPROVE'
            interest_rate_premium = 0.5
        elif default_prob < 0.25:
            risk_tier = 'AA'
            recommendation = 'APPROVE'
            interest_rate_premium = 1.0
        elif default_prob < 0.35:
            risk_tier = 'A'
            recommendation = 'APPROVE'
            interest_rate_premium = 1.5
        elif default_prob < 0.50:
            risk_tier = 'BBB'
            recommendation = 'REVIEW'
            interest_rate_premium = 2.5
        elif default_prob < 0.65:
            risk_tier = 'BB'
            recommendation = 'DECLINE'
            interest_rate_premium = 4.0
        else:
            risk_tier = 'B'
            recommendation = 'DECLINE'
            interest_rate_premium = 6.0
        
        # Top risk factors
        top_risk_factors = []
        if application_dict.get('credit_score', 700) < 600:
            top_risk_factors.append('Low credit score')
        if application_dict.get('debt_to_income_ratio', 0) > 0.5:
            top_risk_factors.append('High debt-to-income ratio')
        if application_dict.get('late_payments_12m', 0) > 2:
            top_risk_factors.append('Multiple late payments')
        if application_dict.get('years_in_business', 5) < 2:
            top_risk_factors.append('Limited business history')
        if application_dict.get('transaction_volatility', 0) > 0.6:
            top_risk_factors.append('High transaction volatility')
        
        return {
            'default_probability': round(float(default_prob), 4),
            'risk_tier': risk_tier,
            'recommendation': recommendation,
            'interest_rate_premium_bps': interest_rate_premium * 100,
            'individual_model_predictions': individual_preds,
            'top_risk_factors': top_risk_factors,
            'confidence': round(abs(default_prob - 0.5) * 2, 4)
        }

    def save(self, path='/home/claude/credit-risk-platform/models/credit_ml_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✅ Model saved → {path}")

    @staticmethod
    def load(path='/home/claude/credit-risk-platform/models/credit_ml_model.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)


# ── Fraud Detection Module ───────────────────────────────────────────────────

class FraudDetectionPipeline:
    """Real-time fraud detection using Isolation Forest + Statistical Rules"""

    def __init__(self):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self.isolation_forest = IsolationForest(
            contamination=0.05, n_estimators=200,
            random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.fraud_rules = []
        self.is_trained = False
        self.thresholds = {}

    def _extract_sequence_features(self, sequences):
        """Convert time-series sequences to feature vectors"""
        features = []
        for seq_data in sequences:
            seq = seq_data['sequence']
            amounts = [d['total_amount'] for d in seq]
            txns = [d['num_transactions'] for d in seq]
            
            # Statistical features
            features.append({
                'mean_amount': np.mean(amounts),
                'std_amount': np.std(amounts),
                'cv_amount': np.std(amounts) / (np.mean(amounts) + 1),
                'max_amount': np.max(amounts),
                'min_amount': np.min(amounts),
                'trend': np.polyfit(range(len(amounts)), amounts, 1)[0],
                'mean_txns': np.mean(txns),
                'std_txns': np.std(txns),
                'zero_days': sum(1 for a in amounts if a < 10),
                'spike_days': sum(1 for a in amounts if a > np.mean(amounts) + 2*np.std(amounts)),
                'last_7_vs_all': np.mean(amounts[-7:]) / (np.mean(amounts) + 1),
                'weekend_ratio': np.mean([d['is_weekend'] for d in seq]),
            })
        return pd.DataFrame(features)

    def train(self, sequences):
        print("\n[FRAUD DETECTION] Training Isolation Forest...")
        features_df = self._extract_sequence_features(sequences)
        X = self.scaler.fit_transform(features_df.fillna(0))
        self.isolation_forest.fit(X)
        
        # Compute threshold stats
        scores = self.isolation_forest.score_samples(X)
        self.thresholds = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'p5': float(np.percentile(scores, 5)),
            'p1': float(np.percentile(scores, 1))
        }
        
        self.is_trained = True
        labeled = [s['fraud_label'] for s in sequences]
        print(f"      Trained on {len(sequences)} sequences | {sum(labeled)} known fraud cases")
        return {'status': 'trained', 'sequences': len(sequences)}

    def detect(self, transaction_sequence):
        """Detect fraud in a transaction sequence"""
        features_df = self._extract_sequence_features([{'sequence': transaction_sequence}])
        X = self.scaler.transform(features_df.fillna(0))
        
        anomaly_score = float(self.isolation_forest.score_samples(X)[0])
        prediction = self.isolation_forest.predict(X)[0]  # -1=anomaly, 1=normal
        
        # Normalized risk score (0-100)
        threshold_range = self.thresholds['mean'] - self.thresholds['p1']
        normalized_score = max(0, min(100, 
            (self.thresholds['mean'] - anomaly_score) / (threshold_range + 0.001) * 100
        ))
        
        # Rule-based checks
        alerts = []
        amounts = [d['total_amount'] for d in transaction_sequence]
        if len(amounts) >= 7:
            if np.mean(amounts[-7:]) < 0.3 * np.mean(amounts[:-7]):
                alerts.append('Sudden revenue drop (>70%) in last 7 days')
            if max(amounts[-7:]) > 5 * np.mean(amounts[:-7]):
                alerts.append('Suspicious transaction spike detected')
        zero_days = sum(1 for a in amounts if a < 10)
        if zero_days > len(amounts) * 0.3:
            alerts.append(f'Excessive zero-activity days: {zero_days}')
        
        return {
            'is_fraud': prediction == -1 or normalized_score > 75,
            'fraud_probability': round(normalized_score / 100, 4),
            'anomaly_score': round(anomaly_score, 4),
            'risk_level': 'HIGH' if normalized_score > 75 else ('MEDIUM' if normalized_score > 40 else 'LOW'),
            'alerts': alerts,
            'requires_review': normalized_score > 50 or len(alerts) > 0
        }


if __name__ == '__main__':
    # Load data
    print("Loading training data...")
    df = pd.read_csv('/home/claude/credit-risk-platform/data/credit_applications.csv')
    
    with open('/home/claude/credit-risk-platform/data/transaction_sequences.json') as f:
        sequences = json.load(f)
    
    # Train credit risk model
    credit_pipeline = CreditRiskMLPipeline()
    metrics = credit_pipeline.train(df)
    credit_pipeline.save()
    
    # Train fraud detection
    fraud_pipeline = FraudDetectionPipeline()
    fraud_pipeline.train(sequences)
    
    with open('/home/claude/credit-risk-platform/models/fraud_pipeline.pkl', 'wb') as f:
        import pickle
        pickle.dump(fraud_pipeline, f)
    
    # Test prediction
    print("\n[TEST] Running sample prediction...")
    sample = {
        'company_name': 'Acme Corp',
        'industry': 'Technology',
        'loan_purpose': 'Business Expansion',
        'annual_revenue': 2_500_000,
        'loan_amount_requested': 500_000,
        'loan_to_revenue_ratio': 0.2,
        'years_in_business': 5,
        'num_employees': 45,
        'credit_score': 710,
        'debt_to_income_ratio': 0.35,
        'monthly_cash_flow': 42_000,
        'existing_loans': 1,
        'late_payments_12m': 0,
        'avg_daily_transactions': 28.5,
        'transaction_volatility': 0.22,
        'max_overdraft_30d': 0
    }
    
    result = credit_pipeline.predict(sample)
    print(f"\n  Application Result:")
    print(f"  Default Probability: {result['default_probability']:.1%}")
    print(f"  Risk Tier:          {result['risk_tier']}")
    print(f"  Recommendation:     {result['recommendation']}")
    print(f"  Rate Premium:       +{result['interest_rate_premium_bps']}bps")
    
    # Save metrics for dashboard
    all_metrics = {
        'credit_risk': metrics,
        'fraud_detection': {'status': 'trained', 'sequences': len(sequences)},
        'sample_prediction': result
    }
    with open('/home/claude/credit-risk-platform/models/training_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("\n✅ All ML models trained and saved!")
