"""Authentication Blueprint"""
from flask import Blueprint, render_template, redirect, url_for, request, flash, current_app
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime
from backend.extensions import db
from backend.models.models import User

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        error_message = None

        if user:
            if not user.is_active:
                error_message = 'Account is inactive'
            elif not user.check_password(password):
                error_message = 'Incorrect password'
            else:
                user.last_login = datetime.utcnow()
                db.session.commit()
                login_user(user, remember=True)
                return redirect(url_for('dashboard.index'))
        else:
            error_message = 'Username not found'

        if current_app.debug:
            flash(error_message, 'error')
        else:
            flash('Invalid credentials', 'error')
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
