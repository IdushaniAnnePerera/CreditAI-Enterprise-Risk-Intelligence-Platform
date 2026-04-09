"""
Enterprise Credit Risk Platform - Module 2: NLP Sentiment Engine
Financial news sentiment analysis using TF-IDF + Logistic Regression
(Production fallback when HuggingFace models are not available)
"""

import numpy as np
import pandas as pd
import re
import json
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder


class FinancialNLPEngine:
    """
    Financial NLP Engine for Credit Risk Sentiment Analysis
    
    Architecture:
    - TF-IDF with financial n-grams + domain lexicon boosting
    - Logistic Regression classifier (production-grade, interpretable)
    - Financial domain lexicon for rule augmentation
    - Designed as drop-in replacement for transformer-based models
    """

    FINANCIAL_POSITIVE_TERMS = {
        'record': 2.0, 'beat': 1.8, 'exceeded': 1.8, 'surpassed': 1.8,
        'upgraded': 2.0, 'growth': 1.5, 'expansion': 1.5, 'profit': 1.5,
        'revenue increase': 2.0, 'strong': 1.5, 'robust': 1.5, 'raised guidance': 2.0,
        'dividend': 1.3, 'buyback': 1.3, 'partnership': 1.3, 'acquisition': 1.2,
        'outperform': 1.8, 'overweight': 1.5, 'buy rating': 2.0, 'cash flow': 1.3,
        'debt free': 2.0, 'fully funded': 1.8, 'approved': 1.5, 'secured funding': 2.0
    }

    FINANCIAL_NEGATIVE_TERMS = {
        'miss': -2.0, 'missed': -2.0, 'decline': -1.8, 'fell': -1.5,
        'downgraded': -2.0, 'loss': -2.0, 'debt': -1.3, 'default': -2.5,
        'bankruptcy': -3.0, 'liquidation': -3.0, 'restructuring': -1.8,
        'investigation': -2.0, 'fraud': -2.5, 'layoffs': -1.8, 'cuts': -1.5,
        'downgrade': -2.0, 'underperform': -1.8, 'sell rating': -2.0,
        'profit warning': -2.0, 'liquidity': -1.5, 'cash burn': -2.0,
        'covenant breach': -2.5, 'creditors': -1.8, 'insolvency': -2.8
    }

    def __init__(self):
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.metrics = {}
        self.class_names = ['negative', 'neutral', 'positive']

    def _preprocess(self, text):
        """Financial text preprocessing"""
        text = text.lower().strip()
        # Normalize financial numbers
        text = re.sub(r'\$[\d,.]+[mbk]?', '[AMOUNT]', text)
        text = re.sub(r'[\d,.]+%', '[PERCENT]', text)
        text = re.sub(r'\b(q[1-4]|fy|h[12])\s*\d{2,4}\b', '[PERIOD]', text)
        # Remove special chars but keep hyphens
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _compute_lexicon_score(self, text):
        """Rule-based lexicon scoring as auxiliary feature"""
        text_lower = text.lower()
        score = 0.0
        for term, weight in self.FINANCIAL_POSITIVE_TERMS.items():
            if term in text_lower:
                score += weight
        for term, weight in self.FINANCIAL_NEGATIVE_TERMS.items():
            if term in text_lower:
                score += weight
        return np.clip(score / 5, -1, 1)

    def train(self, news_df):
        print("\n" + "="*60)
        print("  TRAINING NLP SENTIMENT ENGINE")
        print("="*60)

        texts = news_df['headline'].apply(self._preprocess).tolist()
        labels_raw = news_df['sentiment_label'].tolist()
        
        # Encode labels
        labels = self.label_encoder.fit_transform(labels_raw)
        print(f"\n[1/3] Dataset: {len(texts)} headlines")
        print(f"      Classes: {dict(zip(self.label_encoder.classes_, np.bincount(labels)))}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Build TF-IDF pipeline with financial domain settings
        print("\n[2/3] Building TF-IDF + Logistic Regression pipeline...")
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),         # Capture multi-word financial phrases
                max_features=15000,
                min_df=2,
                sublinear_tf=True,           # Log normalization
                analyzer='word',
                token_pattern=r'\b[a-zA-Z][a-zA-Z\-]+\b'
            )),
            ('clf', LogisticRegression(
                C=1.0, class_weight='balanced',
                max_iter=500, random_state=42, solver='lbfgs'
            ))
        ])

        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring='f1_macro')
        print(f"      5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        print("\n[3/3] Evaluating...")
        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        self.metrics = {
            'accuracy': round(acc, 4),
            'f1_macro': round(f1_macro, 4),
            'cv_f1_mean': round(cv_scores.mean(), 4),
            'cv_f1_std': round(cv_scores.std(), 4),
            'test_samples': len(X_test)
        }
        
        print(f"\n  Accuracy:    {acc:.4f}")
        print(f"  F1 (macro):  {f1_macro:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)}")
        
        self.is_trained = True
        return self.metrics

    def analyze(self, headlines, company_name=None):
        """Analyze sentiment for a list of headlines"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        processed = [self._preprocess(h) for h in headlines]
        probas = self.pipeline.predict_proba(processed)
        predictions = self.pipeline.predict(processed)
        
        results = []
        for i, (h, pred, proba) in enumerate(zip(headlines, predictions, probas)):
            sentiment_label = self.label_encoder.inverse_transform([pred])[0]
            lexicon_score = self._compute_lexicon_score(h)
            
            # Composite credit impact score (-1 to +1)
            neg_p = proba[self.label_encoder.transform(['negative'])[0]]
            pos_p = proba[self.label_encoder.transform(['positive'])[0]]
            credit_impact_score = float(pos_p - neg_p) * 0.7 + lexicon_score * 0.3
            
            results.append({
                'headline': h,
                'sentiment': sentiment_label,
                'confidence': round(float(max(proba)), 4),
                'probabilities': {
                    cls: round(float(p), 4)
                    for cls, p in zip(self.label_encoder.classes_, proba)
                },
                'lexicon_score': round(lexicon_score, 4),
                'credit_impact_score': round(credit_impact_score, 4),
                'credit_signal': 'POSITIVE' if credit_impact_score > 0.2 else (
                    'NEGATIVE' if credit_impact_score < -0.2 else 'NEUTRAL'
                )
            })
        
        # Aggregate summary
        scores = [r['credit_impact_score'] for r in results]
        neg_count = sum(1 for r in results if r['sentiment'] == 'negative')
        pos_count = sum(1 for r in results if r['sentiment'] == 'positive')
        
        summary = {
            'company': company_name or 'Unknown',
            'total_headlines': len(results),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'neutral_count': len(results) - pos_count - neg_count,
            'avg_credit_impact': round(np.mean(scores), 4),
            'sentiment_trend': 'BULLISH' if np.mean(scores) > 0.1 else (
                'BEARISH' if np.mean(scores) < -0.1 else 'STABLE'
            ),
            'credit_adjustment_bps': round(np.mean(scores) * -100, 1),  # Negative = rate increase
            'individual_results': results
        }
        
        return summary

    def save(self, path='/home/claude/credit-risk-platform/models/nlp_engine.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✅ NLP Engine saved → {path}")

    @staticmethod
    def load(path='/home/claude/credit-risk-platform/models/nlp_engine.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)


class NewsAggregator:
    """Aggregates news sentiment signals for credit decisions"""
    
    def __init__(self, nlp_engine):
        self.nlp = nlp_engine
    
    def get_credit_sentiment_report(self, company_name, headlines):
        """Generate a full credit sentiment report"""
        if not headlines:
            return {
                'company': company_name,
                'status': 'NO_DATA',
                'credit_adjustment_bps': 0,
                'recommendation': 'Proceed with standard assessment'
            }
        
        analysis = self.nlp.analyze(headlines, company_name)
        
        # Risk flags
        flags = []
        if analysis['negative_count'] / max(len(headlines), 1) > 0.6:
            flags.append('HIGH_NEGATIVE_NEWS_RATIO')
        if analysis['avg_credit_impact'] < -0.4:
            flags.append('SEVERE_NEGATIVE_SENTIMENT')
        if analysis['sentiment_trend'] == 'BEARISH':
            flags.append('BEARISH_MARKET_SENTIMENT')
        
        return {
            'company': company_name,
            'sentiment_summary': analysis['sentiment_trend'],
            'avg_impact_score': analysis['avg_credit_impact'],
            'credit_adjustment_bps': analysis['credit_adjustment_bps'],
            'risk_flags': flags,
            'headlines_analyzed': len(headlines),
            'breakdown': {
                'positive': analysis['positive_count'],
                'negative': analysis['negative_count'],
                'neutral': analysis['neutral_count']
            },
            'recommendation': self._get_recommendation(analysis, flags)
        }
    
    def _get_recommendation(self, analysis, flags):
        if 'SEVERE_NEGATIVE_SENTIMENT' in flags:
            return 'Manual review required - significant negative news detected'
        elif 'HIGH_NEGATIVE_NEWS_RATIO' in flags:
            return 'Increase rate premium by 50-100bps due to negative news sentiment'
        elif analysis['sentiment_trend'] == 'BULLISH':
            return 'Favorable news sentiment - consider standard or reduced rate'
        return 'Neutral news sentiment - proceed with standard assessment'


if __name__ == '__main__':
    # Load news data
    news_df = pd.read_csv('/home/claude/credit-risk-platform/data/financial_news.csv')
    
    # Train NLP engine
    nlp_engine = FinancialNLPEngine()
    metrics = nlp_engine.train(news_df)
    nlp_engine.save()
    
    # Test analysis
    test_headlines = [
        "TechCorp reports record Q3 earnings, beating analyst expectations by 23%",
        "TechCorp faces regulatory investigation for accounting irregularities",
        "TechCorp announces new CFO as part of strategic transition",
        "TechCorp secures $50M funding round, expanding to Asian markets",
        "TechCorp misses revenue targets, stock falls 18% in after-hours trading"
    ]
    
    aggregator = NewsAggregator(nlp_engine)
    report = aggregator.get_credit_sentiment_report("TechCorp", test_headlines)
    
    print("\n[TEST] Sentiment Analysis Report:")
    print(f"  Company:          {report['company']}")
    print(f"  Sentiment Trend:  {report['sentiment_summary']}")
    print(f"  Avg Impact Score: {report['avg_impact_score']:.4f}")
    print(f"  Rate Adjustment:  {report['credit_adjustment_bps']:+.1f}bps")
    print(f"  Risk Flags:       {report['risk_flags'] or 'None'}")
    print(f"  Recommendation:   {report['recommendation']}")
    
    # Save NLP metrics
    with open('/home/claude/credit-risk-platform/models/nlp_metrics.json', 'w') as f:
        json.dump({'nlp': metrics, 'test_report': report}, f, indent=2)
    
    print("\n✅ NLP Engine complete!")
