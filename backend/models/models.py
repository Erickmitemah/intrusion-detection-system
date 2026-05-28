"""
Database Models for NetGuard IDS
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

# Import from extensions.py — NOT from app.py (avoids circular import)
from backend.extensions import db, login_manager


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='analyst')
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


class NetworkPacket(db.Model):
    __tablename__ = 'network_packets'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    src_ip = db.Column(db.String(45))
    dst_ip = db.Column(db.String(45))
    src_port = db.Column(db.Integer)
    dst_port = db.Column(db.Integer)
    protocol = db.Column(db.String(10))
    packet_size = db.Column(db.Integer)
    duration = db.Column(db.Float)
    flags = db.Column(db.String(50))
    payload_entropy = db.Column(db.Float)
    feature_vector = db.Column(db.Text)
    classification = db.Column(db.String(30), default='unknown')
    risk_score = db.Column(db.Float, default=0.0)
    anomaly_score = db.Column(db.Float, default=0.0)
    cluster_id = db.Column(db.Integer, default=-1)
    is_malicious = db.Column(db.Boolean, default=False)
    alert_id = db.Column(db.Integer, db.ForeignKey('alerts.id'), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'packet_size': self.packet_size,
            'classification': self.classification,
            'risk_score': self.risk_score,
            'anomaly_score': self.anomaly_score,
            'is_malicious': self.is_malicious,
            'cluster_id': self.cluster_id
        }


class Alert(db.Model):
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    alert_type = db.Column(db.String(50))
    severity = db.Column(db.String(20))
    risk_score = db.Column(db.Float)
    source_ip = db.Column(db.String(45))
    destination_ip = db.Column(db.String(45))
    description = db.Column(db.Text)
    detection_method = db.Column(db.String(50))
    status = db.Column(db.String(20), default='open')
    acknowledged_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    acknowledged_at = db.Column(db.DateTime)
    packets = db.relationship('NetworkPacket', backref='alert', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type,
            'severity': self.severity,
            'risk_score': self.risk_score,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'description': self.description,
            'detection_method': self.detection_method,
            'status': self.status,
            'packet_count': len(self.packets)
        }


class MLModel(db.Model):
    __tablename__ = 'ml_models'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    model_type = db.Column(db.String(50))
    version = db.Column(db.String(20))
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    training_samples = db.Column(db.Integer)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    model_path = db.Column(db.String(256))
    dataset_used = db.Column(db.String(100))

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'model_type': self.model_type,
            'version': self.version,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_samples': self.training_samples,
            'training_date': self.training_date.isoformat(),
            'is_active': self.is_active,
            'dataset_used': self.dataset_used
        }


class TrafficStat(db.Model):
    __tablename__ = 'traffic_stats'

    id = db.Column(db.Integer, primary_key=True)
    window_start = db.Column(db.DateTime, index=True)
    window_end = db.Column(db.DateTime)
    total_packets = db.Column(db.Integer, default=0)
    malicious_packets = db.Column(db.Integer, default=0)
    normal_packets = db.Column(db.Integer, default=0)
    unique_src_ips = db.Column(db.Integer, default=0)
    unique_dst_ips = db.Column(db.Integer, default=0)
    avg_packet_size = db.Column(db.Float)
    total_bytes = db.Column(db.Integer, default=0)
    attack_types = db.Column(db.Text)
    protocol_distribution = db.Column(db.Text)


class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    report_type = db.Column(db.String(50))
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    generated_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    period_start = db.Column(db.DateTime)
    period_end = db.Column(db.DateTime)
    content = db.Column(db.Text)
    file_path = db.Column(db.String(256))

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'report_type': self.report_type,
            'generated_at': self.generated_at.isoformat(),
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None
        }
