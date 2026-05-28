"""
Shared Flask extensions — imported by both app.py and models.py
to break the circular import chain.
"""
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()
