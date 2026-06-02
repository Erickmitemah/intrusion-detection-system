"""
NetGuard IDS - Network Intrusion Detection System
Main Flask Application Entry Point
"""

from flask import Flask
from flask_cors import CORS
import os
import logging

def create_app(config_name='development'):
    app = Flask(__name__,
                instance_relative_config=True,
                template_folder='frontend/templates',
                static_folder='frontend/static')

    # Ensure instance folder exists and store the SQLite DB there
    os.makedirs(app.instance_path, exist_ok=True)

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'netguard-ids-secret-key-2024')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL',
        f"sqlite:///{os.path.join(app.instance_path, 'ids_database.db')}"
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

    # Import extensions from shared module (avoids circular imports)
    from backend.extensions import db, login_manager

    # Initialize extensions with this app
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    CORS(app)

    # Setup logging
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/ids.log'),
            logging.StreamHandler()
        ]
    )

    # Import models INSIDE app context so:
    #   1. db.Model has the app bound to it
    #   2. @login_manager.user_loader decorator fires and registers
    with app.app_context():
        from backend.models import models  # noqa: F401

    # Register blueprints
    from backend.routes.auth import auth_bp
    from backend.routes.dashboard import dashboard_bp
    from backend.routes.api import api_bp
    from backend.routes.reports import reports_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(reports_bp, url_prefix='/reports')

    # Create database tables and seed admin user
    with app.app_context():
        db.create_all()
        from backend.utils.seed import seed_admin
        seed_admin(db)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
