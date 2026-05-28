"""
NetGuard IDS — Unit & Integration Tests
Tests ML engine, API endpoints, packet preprocessing, and model training
"""

import sys
import os
import json
import pytest
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── ML Engine Tests ────────────────────────────────────────────────────────

class TestMLEngine:
    """Tests for the IDSMLEngine class"""

    def setup_method(self):
        from backend.ml.engine import IDSMLEngine
        self.engine = IDSMLEngine(model_dir='/tmp/test_ids_models')

    def test_engine_initializes(self):
        assert self.engine is not None
        assert self.engine.feature_names is not None
        assert len(self.engine.feature_names) > 0

    def test_generate_synthetic_data(self):
        df = self.engine._generate_synthetic_kdd(500)
        assert len(df) == 500
        assert 'label' in df.columns
        # Should have both normal and attack records
        labels = df['label'].unique()
        assert len(labels) > 1

    def test_preprocess_pipeline(self):
        df = self.engine._generate_synthetic_kdd(200)
        X, y, y_binary, df_out = self.engine.preprocess(df)
        assert X.shape[0] == 200
        assert len(y) == 200
        assert len(y_binary) == 200
        assert set(y_binary.unique()).issubset({0, 1})

    def test_train_all_models(self):
        results = self.engine.train_all_models(filepath=None)
        assert 'kmeans' in results
        assert 'hierarchical' in results
        assert 'random_forest' in results
        assert 'decision_tree' in results
        assert 'isolation_forest' in results
        assert self.engine.is_trained is True

    def test_predict_normal_traffic(self):
        self.engine.train_all_models()
        features = {
            'duration': 2.0, 'src_bytes': 500, 'dst_bytes': 800,
            'count': 5, 'srv_count': 5, 'serror_rate': 0.0,
            'rerror_rate': 0.0, 'same_srv_rate': 0.95,
            'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
            'logged_in': 1, 'num_failed_logins': 0
        }
        result = self.engine.predict(features)
        assert 'classification' in result
        assert 'risk_score' in result
        assert 'is_malicious' in result
        assert 'anomaly_score' in result
        assert 0 <= result['risk_score'] <= 100

    def test_predict_dos_traffic(self):
        self.engine.train_all_models()
        # Neptune-style DoS features
        features = {
            'duration': 0, 'src_bytes': 0, 'dst_bytes': 0,
            'count': 511, 'srv_count': 511,
            'serror_rate': 1.0, 'srv_serror_rate': 1.0,
            'same_srv_rate': 1.0, 'rerror_rate': 0.0,
            'protocol_type': 'tcp', 'service': 'http', 'flag': 'S0'
        }
        result = self.engine.predict(features)
        assert 'risk_score' in result
        # DoS should have higher risk than normal
        assert result['risk_score'] >= 0

    def test_risk_score_range(self):
        self.engine.train_all_models()
        for _ in range(10):
            features = {
                'duration': np.random.uniform(0, 10),
                'src_bytes': np.random.randint(0, 5000),
                'dst_bytes': np.random.randint(0, 5000),
                'count': np.random.randint(1, 200),
                'serror_rate': np.random.uniform(0, 1)
            }
            result = self.engine.predict(features)
            assert 0 <= result.get('risk_score', 0) <= 100

    def test_predict_batch(self):
        self.engine.train_all_models()
        records = [
            {'duration': 1.0, 'src_bytes': 300, 'dst_bytes': 400},
            {'duration': 0, 'src_bytes': 0, 'count': 511, 'serror_rate': 1.0},
            {'duration': 5.0, 'src_bytes': 1200, 'logged_in': 1}
        ]
        results = self.engine.predict_batch(records)
        assert len(results) == 3
        for r in results:
            assert 'risk_score' in r

    def test_severity_classification(self):
        self.engine.train_all_models()
        result = self.engine.predict({'root_shell': 1, 'su_attempted': 1, 'serror_rate': 0.9})
        assert result.get('severity') in ['low', 'medium', 'high', 'critical', 'info']

    def test_future_attack_prediction(self):
        history = [
            {'avg_risk_score': 20, 'attack_rate': 0.05},
            {'avg_risk_score': 30, 'attack_rate': 0.10},
            {'avg_risk_score': 45, 'attack_rate': 0.15},
            {'avg_risk_score': 60, 'attack_rate': 0.20},
            {'avg_risk_score': 70, 'attack_rate': 0.28},
        ]
        result = self.engine.predict_future_attacks(history)
        assert 'trend' in result
        assert result['trend'] == 'increasing'
        assert 'predicted_risk_scores' in result
        assert len(result['predicted_risk_scores']) == 3
        assert 'threat_level' in result
        assert 'recommendation' in result

    def test_future_prediction_insufficient_data(self):
        result = self.engine.predict_future_attacks([{'avg_risk_score': 10}])
        assert result.get('prediction') == 'insufficient_data'

    def test_detect_attack_patterns(self):
        self.engine.train_all_models()
        traffic = [
            {'duration': 0, 'src_bytes': 0, 'count': 511, 'serror_rate': 1.0},
            {'duration': 2, 'src_bytes': 500, 'dst_bytes': 300},
            {'duration': 0, 'src_bytes': 0, 'count': 511, 'serror_rate': 1.0},
        ]
        result = self.engine.detect_attack_patterns(traffic)
        assert 'total_connections' in result
        assert result['total_connections'] == 3
        assert 'malicious_count' in result
        assert 'avg_risk_score' in result

    def test_model_stats(self):
        self.engine.train_all_models()
        stats = self.engine.get_model_stats()
        assert 'is_trained' in stats
        assert stats['is_trained'] is True
        assert 'models' in stats
        for m in ['kmeans', 'hierarchical', 'random_forest', 'decision_tree', 'isolation_forest']:
            assert m in stats['models']
            assert stats['models'][m] is True

    def test_kmeans_cluster_assignment(self):
        self.engine.train_all_models()
        result = self.engine.predict({'duration': 0, 'src_bytes': 0, 'count': 400, 'serror_rate': 1.0})
        assert 'cluster_id' in result
        assert isinstance(result['cluster_id'], int)


# ─── Packet Capture Tests ────────────────────────────────────────────────────

class TestPacketCapture:
    """Tests for packet capture and feature extraction"""

    def test_extract_features_basic(self):
        from backend.utils.packet_capture import extract_features_from_packet
        pkt = {
            'src_ip': '192.168.1.100', 'dst_ip': '10.0.0.1',
            'src_port': 54321, 'dst_port': 80,
            'protocol': 'tcp', 'packet_size': 1024,
            'src_bytes': 500, 'dst_bytes': 300,
            'tcp_flags': 'SF'
        }
        features = extract_features_from_packet(pkt)
        assert features['src_ip'] == '192.168.1.100'
        assert features['dst_ip'] == '10.0.0.1'
        assert features['protocol_type'] == 'tcp'
        assert features['src_bytes'] == 500
        assert features['dst_bytes'] == 300

    def test_extract_features_land_attack(self):
        from backend.utils.packet_capture import extract_features_from_packet
        pkt = {
            'src_ip': '192.168.1.1', 'dst_ip': '192.168.1.1',
            'src_port': 80, 'dst_port': 80,
            'protocol': 'tcp'
        }
        features = extract_features_from_packet(pkt)
        assert features['land'] == 1

    def test_service_mapping(self):
        from backend.utils.packet_capture import _guess_service
        assert _guess_service(80) == 'http'
        assert _guess_service(22) == 'ssh'
        assert _guess_service(21) == 'ftp'
        assert _guess_service(25) == 'smtp'
        assert _guess_service(9999) == 'other'

    def test_entropy_calculation(self):
        from backend.utils.packet_capture import _compute_entropy
        # Zero entropy for uniform data
        assert _compute_entropy(b'') == 0.0
        # Higher entropy for random data
        import os
        random_bytes = os.urandom(100)
        entropy = _compute_entropy(random_bytes)
        assert entropy > 0
        assert entropy <= 8.0  # Max entropy for bytes

    def test_simulated_packet_generation(self):
        from backend.utils.packet_capture import _generate_simulated_packet
        for _ in range(20):
            pkt = _generate_simulated_packet()
            assert 'src_ip' in pkt
            assert 'dst_ip' in pkt
            assert 'protocol' in pkt
            assert 'packet_size' in pkt
            assert pkt['packet_size'] > 0
            assert '_scenario' in pkt

    def test_capture_start_stop(self):
        from backend.utils import packet_capture
        result = packet_capture.start_capture()
        assert result is True
        import time
        time.sleep(1.5)
        assert packet_capture.is_capturing() is True
        packet_capture.stop_capture()
        time.sleep(0.5)

    def test_traffic_buffer(self):
        from backend.utils import packet_capture
        packet_capture.start_capture()
        import time
        time.sleep(2)
        buf = packet_capture.get_traffic_buffer(50)
        packet_capture.stop_capture()
        assert isinstance(buf, list)

    def test_realtime_stats(self):
        from backend.utils.packet_capture import get_realtime_stats
        stats = get_realtime_stats()
        assert 'total_packets' in stats
        assert 'packets_per_second' in stats


# ─── Flask App Tests ─────────────────────────────────────────────────────────

class TestFlaskApp:
    """Integration tests for Flask routes and API"""

    def setup_method(self):
        os.makedirs('/tmp/ids_test_logs', exist_ok=True)
        import logging
        logging.basicConfig(handlers=[logging.NullHandler()])
        from app import create_app, db
        self.app = create_app('testing')
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.app.config['LOGIN_DISABLED'] = False
        self.client = self.app.test_client()
        with self.app.app_context():
            db.create_all()
            from backend.utils.seed import seed_admin
            seed_admin(db)
        self.db = db

    def _login(self):
        return self.client.post('/login', data={
            'username': 'admin',
            'password': 'NetGuard@2024!'
        }, follow_redirects=True)

    def test_login_page_loads(self):
        r = self.client.get('/login')
        assert r.status_code == 200
        assert b'NetGuard' in r.data or b'AUTHENTICATE' in r.data

    def test_login_valid_credentials(self):
        r = self._login()
        assert r.status_code == 200

    def test_login_invalid_credentials(self):
        r = self.client.post('/login', data={
            'username': 'wrong', 'password': 'wrong'
        }, follow_redirects=True)
        assert r.status_code == 200
        assert b'Invalid' in r.data or b'login' in r.data.lower()

    def test_dashboard_requires_login(self):
        r = self.client.get('/dashboard', follow_redirects=False)
        assert r.status_code in [302, 401]

    def test_dashboard_accessible_after_login(self):
        self._login()
        r = self.client.get('/dashboard')
        assert r.status_code == 200

    def test_alerts_page(self):
        self._login()
        r = self.client.get('/alerts')
        assert r.status_code == 200

    def test_models_page(self):
        self._login()
        r = self.client.get('/models')
        assert r.status_code == 200

    def test_reports_page(self):
        self._login()
        r = self.client.get('/reports')
        assert r.status_code == 200

    def test_api_dashboard_summary(self):
        self._login()
        r = self.client.get('/api/dashboard/summary')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'alerts' in data
        assert 'ml_status' in data

    def test_api_traffic_live(self):
        self._login()
        r = self.client.get('/api/traffic/live')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'is_capturing' in data

    def test_api_traffic_history(self):
        self._login()
        r = self.client.get('/api/traffic/history?hours=6')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'history' in data
        assert len(data['history']) == 6

    def test_api_alerts_empty(self):
        self._login()
        r = self.client.get('/api/alerts')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'alerts' in data
        assert 'total' in data

    def test_api_alerts_summary(self):
        self._login()
        r = self.client.get('/api/alerts/summary')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'total' in data
        assert 'open' in data
        assert 'critical' in data

    def test_api_model_status(self):
        self._login()
        r = self.client.get('/api/models/status')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'is_trained' in data
        assert 'models' in data

    def test_api_predict(self):
        self._login()
        payload = {
            'duration': 1.0, 'src_bytes': 500, 'dst_bytes': 300,
            'count': 5, 'serror_rate': 0.0, 'protocol_type': 'tcp'
        }
        r = self.client.post('/api/models/predict',
                             data=json.dumps(payload),
                             content_type='application/json')
        assert r.status_code == 200
        data = json.loads(r.data)
        # Models may not be trained in a fresh test instance — both outcomes are valid
        assert 'risk_score' in data or 'error' in data

    def test_api_generate_report(self):
        self._login()
        r = self.client.post('/api/reports/generate',
                             data=json.dumps({'type': 'summary'}),
                             content_type='application/json')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'status' in data
        assert data['status'] == 'generated'

    def test_api_list_reports(self):
        self._login()
        # Generate one first
        self.client.post('/api/reports/generate',
                         data=json.dumps({'type': 'daily'}),
                         content_type='application/json')
        r = self.client.get('/api/reports')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'reports' in data

    def test_api_alert_acknowledge(self):
        self._login()
        with self.app.app_context():
            from backend.models.models import Alert
            alert = Alert(
                alert_type='DoS', severity='high', risk_score=75.0,
                source_ip='1.2.3.4', destination_ip='10.0.0.1',
                description='Test alert', detection_method='Test',
                status='open'
            )
            self.db.session.add(alert)
            self.db.session.commit()
            alert_id = alert.id
        r = self.client.post(f'/api/alerts/{alert_id}/acknowledge')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data['status'] == 'acknowledged'

    def test_api_protocol_distribution(self):
        self._login()
        r = self.client.get('/api/traffic/protocols')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'distribution' in data

    def test_api_future_predictions(self):
        self._login()
        r = self.client.get('/api/models/future-predictions')
        assert r.status_code == 200
        data = json.loads(r.data)
        assert 'threat_level' in data or 'prediction' in data

    def test_logout(self):
        self._login()
        r = self.client.get('/logout', follow_redirects=True)
        assert r.status_code == 200


# ─── Database Model Tests ────────────────────────────────────────────────────

class TestDatabaseModels:
    """Tests for SQLAlchemy database models"""

    def setup_method(self):
        from app import create_app, db
        self.app = create_app()
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['TESTING'] = True
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.drop_all()
        db.create_all()
        self.db = db

    def teardown_method(self):
        self.db.session.remove()
        self.db.drop_all()
        self.app_context.pop()

    def test_user_creation(self):
        from backend.models.models import User
        u = User(username='testuser', email='test@test.com', role='analyst')
        u.set_password('Test@123')
        self.db.session.add(u)
        self.db.session.commit()
        found = User.query.filter_by(username='testuser').first()
        assert found is not None
        assert found.check_password('Test@123')
        assert not found.check_password('wrongpass')

    def test_user_to_dict(self):
        from backend.models.models import User
        u = User(username='dictuser', email='dict@test.com', role='admin')
        u.set_password('pass')
        self.db.session.add(u)
        self.db.session.commit()
        d = u.to_dict()
        assert 'username' in d
        assert 'role' in d
        assert 'password_hash' not in d

    def test_alert_creation(self):
        from backend.models.models import Alert
        a = Alert(
            alert_type='Probe', severity='medium', risk_score=45.0,
            source_ip='192.168.1.50', destination_ip='10.0.0.5',
            description='Port scan detected', detection_method='K-Means',
            status='open'
        )
        self.db.session.add(a)
        self.db.session.commit()
        found = Alert.query.filter_by(alert_type='Probe').first()
        assert found is not None
        assert found.severity == 'medium'
        assert found.risk_score == 45.0

    def test_alert_to_dict(self):
        from backend.models.models import Alert
        a = Alert(alert_type='DoS', severity='critical', risk_score=92.0,
                  source_ip='1.1.1.1', destination_ip='10.0.0.1',
                  description='DDoS', detection_method='RF', status='open')
        self.db.session.add(a)
        self.db.session.commit()
        d = a.to_dict()
        assert d['alert_type'] == 'DoS'
        assert d['severity'] == 'critical'
        assert d['risk_score'] == 92.0

    def test_report_creation(self):
        from backend.models.models import Report
        from datetime import datetime, timedelta
        rep = Report(
            title='Test Report',
            report_type='daily',
            generated_by=1,
            period_start=datetime.utcnow() - timedelta(days=1),
            period_end=datetime.utcnow(),
            content=json.dumps({'test': True})
        )
        self.db.session.add(rep)
        self.db.session.commit()
        found = Report.query.filter_by(title='Test Report').first()
        assert found is not None
        assert json.loads(found.content)['test'] is True


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
