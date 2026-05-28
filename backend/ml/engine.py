import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


def _normalize_value(x, minimum=0.0, maximum=1.0):
    if maximum - minimum == 0:
        return 0.0
    return float(max(min((x - minimum) / (maximum - minimum), 1.0), 0.0))


class IDSMLEngine:
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.getcwd()
        self.feature_names = [
            'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
            'serror_rate', 'rerror_rate', 'same_srv_rate', 'logged_in',
            'num_failed_logins', 'root_shell', 'su_attempted'
        ]
        self.categorical_columns = ['protocol_type', 'service', 'flag']
        self.numeric_features = [
            'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
            'serror_rate', 'rerror_rate', 'same_srv_rate', 'logged_in',
            'num_failed_logins', 'root_shell', 'su_attempted'
        ]
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.is_trained = False
        self._dummy_columns: List[str] = []

    def _guess_service(self, port: int) -> str:
        if port in (80, 8080, 443):
            return 'http'
        if port in (22, 2222):
            return 'ssh'
        if port in (53,):
            return 'dns'
        if port in (25, 587):
            return 'smtp'
        if port in (143, 993, 110, 995):
            return 'imap'
        return 'other'

    def _generate_synthetic_kdd(self, n: int):
        rows = []
        for _ in range(n):
            attack = random.random() < 0.25
            duration = abs(random.gauss(2.5 if not attack else 1.0, 4.0))
            count = int(max(1, random.gauss(20 if not attack else 220, 50)))
            srv_count = int(max(1, count * random.uniform(0.6, 1.0)))
            serror_rate = 0.0 if not attack else random.uniform(0.2, 1.0)
            rerror_rate = 0.0 if not attack else random.uniform(0.0, 0.9)
            same_srv_rate = min(1.0, max(0.0, random.gauss(0.9 if not attack else 0.3, 0.2)))
            service = self._guess_service(random.choice([80, 22, 53, 25, 443, 123]))
            flag = random.choice(['SF', 'S0', 'REJ', 'RSTO'])
            label = 'normal' if not attack else random.choice(['dos', 'probe', 'r2l', 'u2r'])
            rows.append({
                'duration': duration,
                'src_bytes': int(abs(random.gauss(300, 600))),
                'dst_bytes': int(abs(random.gauss(400, 800))),
                'count': count,
                'srv_count': srv_count,
                'serror_rate': round(serror_rate, 2),
                'rerror_rate': round(rerror_rate, 2),
                'same_srv_rate': round(same_srv_rate, 2),
                'logged_in': int(not attack or random.random() > 0.5),
                'num_failed_logins': int(0 if not attack else random.randint(0, 5)),
                'root_shell': int(random.random() < 0.02),
                'su_attempted': int(random.random() < 0.01),
                'protocol_type': random.choice(['tcp', 'udp', 'icmp']),
                'service': service,
                'flag': flag,
                'label': label
            })
        return pd.DataFrame(rows)

    def preprocess(self, df: pd.DataFrame):
        if 'label' not in df.columns:
            raise ValueError('Input dataframe must contain a label column.')
        y = df['label']
        y_binary = (y != 'normal').astype(int)
        df = df.copy()
        for col in self.numeric_features:
            if col not in df.columns:
                df[col] = 0
        for col in self.categorical_columns:
            if col not in df.columns:
                df[col] = 'other'
        numeric = df[self.numeric_features].fillna(0.0).astype(float)
        categorical = pd.get_dummies(df[self.categorical_columns].astype(str), dummy_na=False)
        self._dummy_columns = sorted(set(categorical.columns))
        X_numeric = pd.DataFrame(self.scaler.fit_transform(numeric), columns=self.numeric_features)
        X = pd.concat([X_numeric, categorical.reindex(columns=self._dummy_columns, fill_value=0)], axis=1)
        self.feature_columns = X.columns.tolist()
        return X, y, y_binary, df

    def train_all_models(self, filepath: Optional[str] = None):
        df = self._generate_synthetic_kdd(800)
        X, y, y_binary, _ = self.preprocess(df)
        self.models['kmeans'] = KMeans(n_clusters=2, random_state=42).fit(X)
        self.models['hierarchical'] = AgglomerativeClustering(n_clusters=2).fit(X)
        self.models['random_forest'] = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y_binary)
        self.models['decision_tree'] = DecisionTreeClassifier(random_state=42).fit(X, y_binary)
        self.models['isolation_forest'] = IsolationForest(n_estimators=100, contamination=0.2, random_state=42).fit(X)
        self.is_trained = True
        return {
            'kmeans': True,
            'hierarchical': True,
            'random_forest': True,
            'decision_tree': True,
            'isolation_forest': True
        }

    def _build_feature_matrix(self, features: Dict[str, Any]):
        record = {k: features.get(k, 0) for k in self.numeric_features}
        record['protocol_type'] = str(features.get('protocol_type', 'tcp'))
        record['service'] = str(features.get('service', self._guess_service(int(features.get('dst_port', 0)))))
        record['flag'] = str(features.get('flag', 'SF'))
        df = pd.DataFrame([record])
        numeric = df[self.numeric_features].fillna(0.0).astype(float)
        X_numeric = pd.DataFrame(self.scaler.transform(numeric), columns=self.numeric_features)
        categorical = pd.get_dummies(df[self.categorical_columns].astype(str), dummy_na=False)
        for col in self._dummy_columns:
            if col not in categorical.columns:
                categorical[col] = 0
        categorical = categorical.reindex(columns=self._dummy_columns, fill_value=0)
        return pd.concat([X_numeric, categorical], axis=1)

    def predict(self, features: Dict[str, Any]):
        if not self.is_trained:
            self.train_all_models()
        X = self._build_feature_matrix(features)
        result: Dict[str, Any] = {
            'classification': 'normal',
            'risk_score': 0.0,
            'is_malicious': False,
            'anomaly_score': 0.0,
            'cluster_id': 0,
            'severity': 'info',
            'detection_method': 'ML Ensemble'
        }
        try:
            pred = self.models['random_forest'].predict(X)[0]
            is_malicious = bool(pred)
            anomaly_raw = self.models['isolation_forest'].decision_function(X)[0]
            cluster_id = int(self.models['kmeans'].predict(X)[0])
            base_score = float(abs(X[self.numeric_features].values).sum() / (len(self.numeric_features) * 10))
            risk_score = min(100.0, max(0.0, 50.0 if is_malicious else 15.0 + base_score))
            severity = 'high' if risk_score >= 75 else 'medium' if risk_score >= 40 else 'low'
            if features.get('serror_rate', 0) > 0.5 or features.get('count', 0) > 200:
                severity = 'critical' if risk_score >= 50 else 'high'
            classification = 'dos' if features.get('count', 0) > 200 or features.get('serror_rate', 0.5) else 'normal'
            result.update({
                'classification': classification,
                'risk_score': round(risk_score, 2),
                'is_malicious': is_malicious,
                'anomaly_score': float(_normalize_value(-anomaly_raw, -1.0, 0.0)),
                'cluster_id': cluster_id,
                'severity': severity
            })
        except Exception:
            result.update({
                'classification': 'unknown',
                'risk_score': 0.0,
                'is_malicious': False,
                'anomaly_score': 0.0,
                'cluster_id': 0,
                'severity': 'info'
            })
        return result

    def predict_batch(self, records: List[Dict[str, Any]]):
        return [self.predict(record) for record in records]

    def predict_future_attacks(self, history: List[Dict[str, Any]]):
        if len(history) < 2:
            return {'prediction': 'insufficient_data'}
        values = [h.get('avg_risk_score', 0) for h in history if isinstance(h.get('avg_risk_score', 0), (int, float))]
        if len(values) < 2:
            return {'prediction': 'insufficient_data'}
        trend = 'increasing' if values[-1] > values[-2] else 'decreasing' if values[-1] < values[-2] else 'stable'
        predicted = [round(values[-1] * factor, 2) for factor in (1.1, 1.2, 1.3)]
        threat_level = 'critical' if values[-1] > 65 else 'high' if values[-1] > 45 else 'medium'
        return {
            'trend': trend,
            'predicted_risk_scores': predicted,
            'threat_level': threat_level,
            'recommendation': 'Increase monitoring and validate firewall rules.'
        }

    def detect_attack_patterns(self, traffic: List[Dict[str, Any]]):
        total_connections = len(traffic)
        malicious_count = 0
        total_risk = 0.0
        for pkt in traffic:
            features = self.predict(pkt)
            if features.get('is_malicious'):
                malicious_count += 1
            total_risk += features.get('risk_score', 0)
        return {
            'total_connections': total_connections,
            'malicious_count': malicious_count,
            'avg_risk_score': round(total_risk / max(total_connections, 1), 2),
            'detected_patterns': ['dos' if pkt.get('count', 0) > 200 else 'normal' for pkt in traffic]
        }

    def get_model_stats(self):
        return {
            'is_trained': self.is_trained,
            'models': {k: bool(v) for k, v in self.models.items()}
        }


_instance: Optional[IDSMLEngine] = None


def get_ml_engine():
    global _instance
    if _instance is None:
        _instance = IDSMLEngine()
        _instance.train_all_models()
    return _instance
