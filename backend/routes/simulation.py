"""
NetGuard IDS — Simulation API Blueprint
REST endpoints for the attack simulation lab.
"""

import time
import json
import threading
from collections import Counter
from datetime import datetime

from flask import Blueprint, jsonify, request
from flask_login import login_required

from backend.ml.engine import get_ml_engine
from backend.routes.snapshots import auto_snapshot_before_sim
from simulation.attack_simulator import (
    generate_attack_wave, generate_mixed_scenario,
    ATTACK_PROFILES, get_all_attack_names
)

sim_bp = Blueprint('simulation', __name__)
ml_engine = get_ml_engine()

# Shared evaluation state so the browser can poll progress
_eval_state = {'running': False, 'result': None, 'error': None}


# ── Page route ────────────────────────────────────────────────────────────────
@sim_bp.route('/simulation')
@login_required
def simulation_page():
    from flask import render_template
    return render_template('simulation.html')


# ── API: run single attack ────────────────────────────────────────────────────
@sim_bp.route('/api/simulation/run', methods=['POST'])
@login_required
def run_simulation():
    """
    Simulate a single attack type and return per-packet detection results.
    Body: { "attack_name": "neptune", "n_packets": 50 }
    """
    data = request.json or {}
    attack_name = data.get('attack_name', 'neptune')
    n_packets   = min(int(data.get('n_packets', 50)), 500)

    if attack_name not in ATTACK_PROFILES:
        return jsonify({'error': f'Unknown attack: {attack_name}'}), 400

    if not ml_engine.is_trained:
        # Auto-train if not done yet
        ml_engine.train_all_models()

    # Auto-snapshot BEFORE simulation so model state can be restored afterwards
    snap_name = auto_snapshot_before_sim(attack_name)

    profile  = ATTACK_PROFILES[attack_name]
    category = profile['category']
    packets  = generate_attack_wave(attack_name, n_packets)

    detected     = 0
    risk_scores  = []
    packet_results = []
    class_counts = Counter()
    t0 = time.perf_counter()

    for pkt in packets:
        result = ml_engine.predict(pkt)
        risk   = result.get('risk_score', 0)
        is_mal = result.get('is_malicious', False)
        pred   = result.get('classification', 'unknown')

        risk_scores.append(round(risk, 2))
        class_counts[result.get('classification', 'unknown')] += 1

        # Detection logic
        is_detected = (not is_mal) if category == 'normal' else is_mal

        if is_detected:
            detected += 1

        packet_results.append({
            'classification': pred,
            'risk_score':     round(risk, 2),
            'is_malicious':   is_mal,
            'anomaly_score':  round(result.get('anomaly_score', 0), 4),
            'cluster_id':     result.get('cluster_id', -1),
            'severity':       result.get('severity', 'info'),
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Map predicted classes to categories for distribution chart
    cat_map = {'normal':'normal','DoS':'DoS','Probe':'Probe','R2L':'R2L','U2R':'U2R'}
    cat_dist = Counter()
    for cls, cnt in class_counts.items():
        cat_dist[cat_map.get(cls, 'normal')] += cnt

    return jsonify({
        'attack_name':       attack_name,
        'category':          category,
        'severity':          profile['severity'],
        'description':       profile['description'],
        'total_packets':     n_packets,
        'detected':          detected,
        'missed':            n_packets - detected,
        'detection_rate':    round(detected / n_packets * 100, 2),
        'avg_risk':          round(sum(risk_scores) / len(risk_scores), 2),
        'ms_per_packet':     round(elapsed_ms / n_packets, 3),
        'total_ms':          round(elapsed_ms, 2),
        'risk_scores':       risk_scores,
        'class_distribution': dict(cat_dist),
        'packet_results':    packet_results[:100],  # cap at 100 for response size
        'snapshot_name':     snap_name,
    })


# ── API: run mixed scenario ────────────────────────────────────────────────────
@sim_bp.route('/api/simulation/scenario', methods=['POST'])
@login_required
def run_scenario():
    """
    Run a pre-defined mixed-traffic scenario.
    Body: { "scenario": "realistic" }
    """
    data     = request.json or {}
    scenario = data.get('scenario', 'realistic')

    if not ml_engine.is_trained:
        ml_engine.train_all_models()

    # Auto-snapshot before scenario so model can be restored
    snap_name = auto_snapshot_before_sim(f'scenario_{scenario}')

    packets = generate_mixed_scenario(scenario)
    total   = len(packets)

    correct = 0
    malicious_detected = 0
    total_malicious    = 0
    risk_scores  = []
    class_counts = Counter()
    t0 = time.perf_counter()

    for pkt in packets:
        result    = ml_engine.predict(pkt)
        true_cat  = pkt.get('true_category', 'normal')
        pred      = result.get('classification', 'unknown')
        is_mal    = result.get('is_malicious', False)
        risk      = result.get('risk_score', 0)

        risk_scores.append(risk)
        class_counts[pred] += 1

        if true_cat != 'normal':
            total_malicious += 1
            if is_mal:
                malicious_detected += 1

        # Correct if: true normal & predicted not malicious, OR true attack & detected
        if (true_cat == 'normal' and not is_mal) or \
           (true_cat != 'normal' and is_mal):
            correct += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    dr  = malicious_detected / max(total_malicious, 1) * 100
    acc = correct / total * 100
    fpr_denom = max(total - total_malicious, 1)
    false_positives = sum(
        1 for pkt in packets
        if pkt.get('true_category') == 'normal' and
           ml_engine.predict(pkt).get('is_malicious', False)
    )
    fpr = false_positives / fpr_denom * 100

    cat_map  = {'normal':'normal','DoS':'DoS','Probe':'Probe','R2L':'R2L','U2R':'U2R'}
    cat_dist = Counter()
    for cls, cnt in class_counts.items():
        cat_dist[cat_map.get(cls, 'normal')] += cnt

    return jsonify({
        'scenario':            scenario,
        'total_packets':       total,
        'total_malicious':     total_malicious,
        'malicious_detected':  malicious_detected,
        'detection_rate':      round(dr, 2),
        'accuracy':            round(acc, 2),
        'false_positive_rate': round(fpr, 2),
        'avg_risk':            round(sum(risk_scores) / len(risk_scores), 2),
        'ms_per_packet':       round(elapsed_ms / total, 3),
        'class_distribution':  dict(cat_dist),
        'snapshot_name':     snap_name,
    })


# ── API: full evaluation (async) ──────────────────────────────────────────────
@sim_bp.route('/api/simulation/evaluate', methods=['POST'])
@login_required
def full_evaluate():
    """
    Runs the complete ModelEvaluator pipeline (trains + simulates all attacks).
    Runs in a background thread; result is returned synchronously for simplicity
    (with a generous timeout on the JS side).
    """
    global _eval_state
    data = request.json or {}
    packets_per_attack = min(int(data.get('packets_per_attack', 80)), 200)

    if _eval_state['running']:
        return jsonify({'error': 'Evaluation already running'}), 409

    _eval_state = {'running': True, 'result': None, 'error': None}

    try:
        from simulation.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        summary   = evaluator.run(packets_per_attack=packets_per_attack)
        _eval_state['result']  = summary
        _eval_state['running'] = False
        return jsonify(summary)
    except Exception as e:
        _eval_state['error']   = str(e)
        _eval_state['running'] = False
        return jsonify({'error': str(e)}), 500


# ── API: evaluation status ────────────────────────────────────────────────────
@sim_bp.route('/api/simulation/status')
@login_required
def eval_status():
    return jsonify({
        'running': _eval_state['running'],
        'has_result': _eval_state['result'] is not None,
        'error': _eval_state['error'],
    })


# ── API: list available attacks ───────────────────────────────────────────────
@sim_bp.route('/api/simulation/attacks')
@login_required
def list_attacks():
    attacks = []
    for name, profile in ATTACK_PROFILES.items():
        attacks.append({
            'name':        name,
            'category':    profile['category'],
            'severity':    profile['severity'],
            'description': profile['description'],
        })
    return jsonify({'attacks': attacks})


# ── API: quick accuracy benchmark ─────────────────────────────────────────────
@sim_bp.route('/api/simulation/accuracy', methods=['POST'])
@login_required
def accuracy_benchmark():
    """
    Quick accuracy benchmark: 30 packets per attack type, return per-model metrics.
    """
    if not ml_engine.is_trained:
        ml_engine.train_all_models()

    n = 30
    results = {}

    for attack_name, profile in ATTACK_PROFILES.items():
        packets  = generate_attack_wave(attack_name, n)
        category = profile['category']
        detected = 0
        risks    = []

        for pkt in packets:
            result = ml_engine.predict(pkt)
            is_mal = result.get('is_malicious', False)
            risks.append(result.get('risk_score', 0))
            if category == 'normal':
                if not is_mal:
                    detected += 1
            else:
                if is_mal:
                    detected += 1

        results[attack_name] = {
            'category':       category,
            'detection_rate': round(detected / n * 100, 1),
            'avg_risk':       round(sum(risks) / len(risks), 1),
            'severity':       profile['severity'],
        }

    # Summary by category
    by_cat = {}
    for name, r in results.items():
        cat = r['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r['detection_rate'])

    category_summary = {
        cat: round(sum(rates) / len(rates), 1)
        for cat, rates in by_cat.items()
    }

    overall = round(
        sum(r['detection_rate'] for name, r in results.items() if r['category'] != 'normal') /
        max(sum(1 for r in results.values() if r['category'] != 'normal'), 1), 1
    )

    return jsonify({
        'per_attack':        results,
        'category_summary':  category_summary,
        'overall_detection': overall,
        'model_status':      ml_engine.get_model_stats()['is_trained'],
    })
