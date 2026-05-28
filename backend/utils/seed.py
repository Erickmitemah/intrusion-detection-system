"""Seed initial admin user"""
from datetime import datetime


def seed_admin(db):
    try:
        from backend.models.models import User
        if User.query.filter_by(username='admin').first():
            return
        admin = User(
            username='admin',
            email='admin@netguard.ids',
            role='admin',
            is_active=True
        )
        admin.set_password('NetGuard@2024!')
        db.session.add(admin)
        db.session.commit()
        print("✓ Admin user created: admin / NetGuard@2024!")
    except Exception as e:
        print(f"Seed error (may already exist): {e}")
