"""
NetGuard IDS — Model Snapshot API
Endpoints to save, restore, list and delete ML model snapshots.
Automatically takes a snapshot before every simulation run.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required
from backend.ml.engine import get_ml_engine, get_snapshot_manager

snap_bp = Blueprint('snapshots', __name__)


def _engine():
    return get_ml_engine()

def _mgr():
    return get_snapshot_manager()


# ── List all snapshots ────────────────────────────────────────────────────────
@snap_bp.route('/api/snapshots')
@login_required
def list_snapshots():
    snaps = _mgr().list_snapshots()
    return jsonify({'snapshots': snaps, 'count': len(snaps)})


# ── Save a new snapshot ───────────────────────────────────────────────────────
@snap_bp.route('/api/snapshots/save', methods=['POST'])
@login_required
def save_snapshot():
    """
    Body: { "name": "baseline_v1", "description": "...", "tags": ["pre-sim"] }
    """
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400

    # Sanitise: only alphanumeric, hyphens, underscores
    import re
    if not re.match(r'^[\w\-]+$', name):
        return jsonify({'error': 'name may only contain letters, digits, hyphens and underscores'}), 400

    engine = _engine()
    if not engine.is_trained:
        return jsonify({'error': 'Models are not trained yet. Train them first.'}), 409

    try:
        meta = _mgr().save(
            engine,
            name=name,
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )
        return jsonify({'status': 'saved', 'snapshot': meta})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Restore a snapshot ────────────────────────────────────────────────────────
@snap_bp.route('/api/snapshots/restore', methods=['POST'])
@login_required
def restore_snapshot():
    """
    Body: { "name": "baseline_v1" }
    """
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400

    try:
        meta = _mgr().restore(_engine(), name)
        return jsonify({'status': 'restored', 'snapshot': meta})
    except KeyError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Delete a snapshot ─────────────────────────────────────────────────────────
@snap_bp.route('/api/snapshots/delete', methods=['POST'])
@login_required
def delete_snapshot():
    """
    Body: { "name": "baseline_v1" }
    """
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400

    # Prevent deleting protected snapshots
    if name.startswith('auto_presim_'):
        pass  # auto snapshots can always be deleted

    ok = _mgr().delete(name)
    if ok:
        return jsonify({'status': 'deleted', 'name': name})
    return jsonify({'error': f"Snapshot '{name}' not found"}), 404


# ── Get single snapshot metadata ──────────────────────────────────────────────
@snap_bp.route('/api/snapshots/<name>')
@login_required
def get_snapshot(name):
    meta = _mgr().get_meta(name)
    if not meta:
        return jsonify({'error': f"Snapshot '{name}' not found"}), 404
    return jsonify({'snapshot': meta})


# ── Auto-snapshot helper (called by simulation routes) ────────────────────────
def auto_snapshot_before_sim(label: str = 'simulation') -> str:
    """
    Creates an automatic snapshot named  auto_presim_<label>_<timestamp>.
    Returns the snapshot name so the caller can surface it to the user.
    Called automatically before every simulation run.
    """
    from datetime import datetime
    engine = _engine()
    if not engine.is_trained:
        return None

    ts   = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    name = f'auto_presim_{label}_{ts}'
    try:
        _mgr().save(
            engine,
            name=name,
            description=f'Auto-snapshot before {label} simulation',
            tags=['auto', 'pre-simulation', label]
        )
        return name
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Auto-snapshot failed: {e}")
        return None
