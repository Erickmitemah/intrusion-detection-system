"""
NetGuard IDS - REST API Blueprint
All JSON API endpoints for the frontend dashboard
"""

import json
import threading
import random
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from backend.extensions import db
from backend.models.models import Alert, NetworkPacket, MLModel, TrafficStat, Report
from backend.ml.engine import get_ml_engine
from backend.utils.packet_capture import (
    start_capture, stop_capture, get_traffic_buffer,
    get_realtime_stats, is_capturing, extract_features_from_packet,
    _generate_simulated_packet
)

api_bp = Blueprint('api', __name__)
ml_engine = get_ml_engine()

# ── In-memory state for live capture ──────────────────────────────────────

_recent_predictions = []
_alert_callbacks = []


def _on_packet(features, scenario):
    """Called by capture loop for each packet. Run ML, store alert if needed."""
    global _recent_predictions
    try:
        result = ml_engine.predict(features)
        result['timestamp'] = datetime.utcnow().isoformat()
        result['src_ip'] = features.get('src_ip')
        result['dst_ip'] = features.get('dst_ip')
        result['protocol'] = features.get('protocol_type', 'tcp')

        _recent_predictions.append(result)
        if len(_recent_predictions) > 500:
            _recent_predictions = _recent_predictions[-500:]

        if result.get('is_malicious') and result.get('risk_score', 0) >= 50:
            _create_alert_from_prediction(result, features)
    except Exception as e:
        pass


def _create_alert_from_prediction(result, features):
    """Persist an alert to the database"""
    try:
        # Use a quick background DB write
        severity_map = {'critical': 'critical', 'high': 'high', 'medium': 'medium', 'low': 'low', 'info': 'info'}
        alert = Alert(
            alert_type=result.get('classification', 'unknown'),
            severity=result.get('severity', 'medium'),
            risk_score=result.get('risk_score', 0),
            source_ip=features.get('src_ip', 'unknown'),
            destination_ip=features.get('dst_ip', 'unknown'),
            description=f"Detected {result.get('classification')} attack from {features.get('src_ip')}. "
                        f"Risk Score: {result.get('risk_score'):.1f}. "
                        f"Method: {result.get('detection_method', 'ML Ensemble')}",
            detection_method='ML Ensemble (RF + IF + KMeans)',
            status='open'
        )
        db.session.add(alert)
        db.session.commit()
    except Exception:
        pass


# ── Traffic Endpoints ──────────────────────────────────────────────────────

@api_bp.route('/traffic/live')
@login_required
def traffic_live():
    """Live traffic stats and recent predictions"""
    stats = get_realtime_stats()
    recent = _recent_predictions[-50:]

    malicious_count = sum(1 for p in recent if p.get('is_malicious'))
    attack_types = {}
    for p in recent:
        cls = p.get('classification', 'unknown')
        if cls != 'normal':
            attack_types[cls] = attack_types.get(cls, 0) + 1

    return jsonify({
        'is_capturing': is_capturing(),
        'stats': stats,
        'recent_predictions': recent[-20:],
        'malicious_count': malicious_count,
        'normal_count': len(recent) - malicious_count,
        'attack_types': attack_types,
        'total_analyzed': len(_recent_predictions)
    })


@api_bp.route('/traffic/start', methods=['POST'])
@login_required
def start_traffic_capture():
    if is_capturing():
        return jsonify({'status': 'already_running'})
    start_capture(callback=_on_packet)
    return jsonify({'status': 'started'})


@api_bp.route('/traffic/stop', methods=['POST'])
@login_required
def stop_traffic_capture():
    stop_capture()
    return jsonify({'status': 'stopped'})


@api_bp.route('/traffic/history')
@login_required
def traffic_history():
    """Simulated hourly traffic history for charts"""
    hours = int(request.args.get('hours', 24))
    history = []
    base_time = datetime.utcnow() - timedelta(hours=hours)

    for i in range(hours):
        t = base_time + timedelta(hours=i)
        # Simulate realistic traffic pattern (higher during business hours)
        hour = t.hour
        multiplier = 1.5 if 8 <= hour <= 18 else 0.6
        total = int(random.gauss(500, 80) * multiplier)
        malicious = int(total * random.uniform(0.02, 0.15))
        history.append({
            'timestamp': t.isoformat(),
            'total_packets': total,
            'malicious': malicious,
            'normal': total - malicious,
            'avg_risk': round(random.uniform(10, 45) + (malicious / total) * 60, 2)
        })

    return jsonify({'history': history})


@api_bp.route('/traffic/protocols')
@login_required
def protocol_distribution():
    """Protocol distribution for pie chart"""
    return jsonify({
        'distribution': {
            'TCP': random.randint(55, 70),
            'UDP': random.randint(15, 25),
            'ICMP': random.randint(5, 10),
            'HTTP': random.randint(10, 20),
            'DNS': random.randint(3, 8)
        }
    })


# ── Alert Endpoints ────────────────────────────────────────────────────────

@api_bp.route('/alerts')
@login_required
def get_alerts():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    severity = request.args.get('severity')
    status = request.args.get('status')

    query = Alert.query
    if severity:
        query = query.filter_by(severity=severity)
    if status:
        query = query.filter_by(status=status)

    alerts = query.order_by(Alert.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({
        'alerts': [a.to_dict() for a in alerts.items],
        'total': alerts.total,
        'pages': alerts.pages,
        'current_page': page
    })


@api_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    alert.status = 'investigating'
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.utcnow()
    db.session.commit()
    return jsonify({'status': 'acknowledged', 'alert': alert.to_dict()})


@api_bp.route('/alerts/<int:alert_id>/resolve', methods=['POST'])
@login_required
def resolve_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    alert.status = 'resolved'
    db.session.commit()
    return jsonify({'status': 'resolved'})


@api_bp.route('/alerts/summary')
@login_required
def alerts_summary():
    total = Alert.query.count()
    open_alerts = Alert.query.filter_by(status='open').count()
    critical = Alert.query.filter_by(severity='critical').count()
    high = Alert.query.filter_by(severity='high').count()
    today = Alert.query.filter(Alert.timestamp >= datetime.utcnow().replace(hour=0, minute=0)).count()

    return jsonify({
        'total': total,
        'open': open_alerts,
        'critical': critical,
        'high': high,
        'today': today,
        'resolved': Alert.query.filter_by(status='resolved').count()
    })


# ── ML Model Endpoints ─────────────────────────────────────────────────────

@api_bp.route('/models/status')
@login_required
def model_status():
    stats = ml_engine.get_model_stats()
    return jsonify(stats)


@api_bp.route('/models/train', methods=['POST'])
@login_required
def train_models():
    """Trigger model training (async)"""
    def _train():
        try:
            filepath = request.json.get('dataset_path') if request.json else None
            ml_engine.train_all_models(filepath)
        except Exception as e:
            pass

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return jsonify({'status': 'training_started', 'message': 'Model training initiated in background'})


@api_bp.route('/models/predict', methods=['POST'])
@login_required
def predict_single():
    """Predict on a single connection feature set"""
    features = request.json or {}
    result = ml_engine.predict(features)
    return jsonify(result)


@api_bp.route('/models/metrics')
@login_required
def model_metrics():
    stats = ml_engine.get_model_stats()
    return jsonify({
        'metrics': stats.get('metrics', {}),
        'is_trained': stats.get('is_trained', False)
    })


@api_bp.route('/models/plots')
@login_required
def model_plots():
    """Return list of available plot images"""
    import os
    plot_dir = 'frontend/static/assets/plots'
    plots = []
    if os.path.exists(plot_dir):
        for f in os.listdir(plot_dir):
            if f.endswith('.png'):
                plots.append({
                    'name': f.replace('_', ' ').replace('.png', '').title(),
                    'url': f'/static/assets/plots/{f}',
                    'filename': f
                })
    return jsonify({'plots': plots})


@api_bp.route('/models/future-predictions')
@login_required
def future_predictions():
    """Predict future attack likelihood based on recent trends"""
    # Build historical stats from recent predictions
    history = []
    if _recent_predictions:
        window_size = 20
        for i in range(0, min(100, len(_recent_predictions)), window_size):
            window = _recent_predictions[i:i + window_size]
            if window:
                malicious = sum(1 for p in window if p.get('is_malicious'))
                avg_risk = sum(p.get('risk_score', 0) for p in window) / len(window)
                history.append({
                    'avg_risk_score': avg_risk,
                    'attack_rate': malicious / len(window)
                })
    else:
        # Simulated history
        history = [{'avg_risk_score': random.uniform(20, 60), 'attack_rate': random.uniform(0.05, 0.25)} for _ in range(6)]

    prediction = ml_engine.predict_future_attacks(history)
    return jsonify(prediction)


# ── Dashboard Summary ──────────────────────────────────────────────────────

@api_bp.route('/dashboard/summary')
@login_required
def dashboard_summary():
    """Main dashboard summary endpoint"""
    alert_summary = {
        'total': Alert.query.count(),
        'open': Alert.query.filter_by(status='open').count(),
        'critical': Alert.query.filter_by(severity='critical').count(),
        'high': Alert.query.filter_by(severity='high').count(),
        'resolved': Alert.query.filter_by(status='resolved').count(),
        'today': Alert.query.filter(Alert.timestamp >= datetime.utcnow().replace(hour=0, minute=0)).count()
    }

    recent_alerts = [a.to_dict() for a in Alert.query.order_by(Alert.timestamp.desc()).limit(5).all()]

    # Simulated top attackers
    top_attackers = [
        {'ip': f"45.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
         'count': random.randint(50, 500), 'type': random.choice(['DoS', 'Probe', 'R2L'])}
        for _ in range(5)
    ]

    return jsonify({
        'alerts': alert_summary,
        'recent_alerts': recent_alerts,
        'ml_status': ml_engine.get_model_stats(),
        'capture_active': is_capturing(),
        'top_attackers': top_attackers,
        'risk_level': 'high' if alert_summary['critical'] > 0 else 'medium' if alert_summary['high'] > 0 else 'low'
    })


# ── Reports Endpoints ──────────────────────────────────────────────────────

@api_bp.route('/reports/generate', methods=['POST'])
@login_required
def generate_report():
    data = request.json or {}
    report_type = data.get('type', 'summary')

    report_data = {
        'generated_at': datetime.utcnow().isoformat(),
        'type': report_type,
        'alerts_summary': {
            'total': Alert.query.count(),
            'critical': Alert.query.filter_by(severity='critical').count(),
            'resolved': Alert.query.filter_by(status='resolved').count()
        },
        'ml_metrics': ml_engine.get_model_stats().get('metrics', {}),
        'recommendations': [
            'Implement rate limiting on suspicious source IPs',
            'Update firewall rules to block identified attack patterns',
            'Schedule regular model retraining with new traffic data',
            'Enable two-factor authentication for all admin accounts'
        ]
    }

    report = Report(
        title=f"{report_type.title()} Security Report - {datetime.utcnow().strftime('%Y-%m-%d')}",
        report_type=report_type,
        generated_by=current_user.id,
        period_start=datetime.utcnow() - timedelta(days=7),
        period_end=datetime.utcnow(),
        content=json.dumps(report_data)
    )
    db.session.add(report)
    db.session.commit()

    return jsonify({'status': 'generated', 'report': report.to_dict(), 'data': report_data})


@api_bp.route('/reports')
@login_required
def list_reports():
    reports = Report.query.order_by(Report.generated_at.desc()).limit(20).all()
    return jsonify({'reports': [r.to_dict() for r in reports]})
