"""
NetGuard IDS — Model Evaluator
Runs all attack simulations through every ML model and produces
a full accuracy report: per-class metrics, confusion matrices,
detection rates, false positive rates, and timing benchmarks.
"""

import sys
import os
import time
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.attack_simulator import (
    ATTACK_PROFILES, generate_attack_wave, generate_mixed_scenario,
    get_all_attack_names
)
from backend.ml.engine import IDSMLEngine

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)


# ─── Colour Palette ────────────────────────────────────────────────────────────
C = {
    'bg':      '#020408',
    'surface': '#0d1520',
    'border':  '#1a2840',
    'accent':  '#00d4ff',
    'green':   '#00ff88',
    'red':     '#ff3d71',
    'yellow':  '#ffcc00',
    'orange':  '#ff8c00',
    'purple':  '#b44fff',
    'text':    '#c8d8e8',
    'dim':     '#4a6080',
}

SEV_COLOUR = {
    'normal': C['green'],
    'DoS':    C['red'],
    'Probe':  C['yellow'],
    'R2L':    C['orange'],
    'U2R':    C['purple'],
}

ATTACK_COLOURS = [C['accent'], C['green'], C['red'], C['yellow'], C['orange'], C['purple']]


class ModelEvaluator:
    """
    Trains all NetGuard models on synthetic KDD data, then runs every
    simulated attack type through them and measures detection performance.
    """

    def __init__(self):
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║     NetGuard IDS — Attack Simulation & Evaluation    ║")
        print("╚══════════════════════════════════════════════════════╝\n")

        self.engine = IDSMLEngine(model_dir='backend/ml/saved_models')
        os.makedirs('simulation/results', exist_ok=True)
        os.makedirs('frontend/static/assets/plots', exist_ok=True)

        self.results = {}          # per-attack detection results
        self.all_true   = []       # ground-truth labels
        self.all_pred_rf = []      # RF predictions
        self.all_pred_dt = []      # DT predictions
        self.all_pred_if = []      # Isolation Forest binary predictions
        self.timing = {}           # inference timing per model

    # ─── Step 1: Train ────────────────────────────────────────────────────────

    def train(self):
        print("[ STEP 1 ]  Training all ML models on KDD-99 dataset...")
        t0 = time.time()
        metrics = self.engine.train_all_models()
        elapsed = time.time() - t0
        print(f"            ✓ Training complete in {elapsed:.1f}s\n")

        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  Model              Metric          Score       │")
        print("  ├─────────────────────────────────────────────────┤")
        rf = metrics.get('random_forest', {})
        dt = metrics.get('decision_tree', {})
        km = metrics.get('kmeans', {})
        hc = metrics.get('hierarchical', {})
        iso= metrics.get('isolation_forest', {})
        print(f"  │  Random Forest      Accuracy        {rf.get('accuracy',0)*100:6.2f}%    │")
        print(f"  │  Random Forest      F1-Score        {rf.get('f1_score',0)*100:6.2f}%    │")
        print(f"  │  Decision Tree      Accuracy        {dt.get('accuracy',0)*100:6.2f}%    │")
        print(f"  │  Decision Tree      F1-Score        {dt.get('f1_score',0)*100:6.2f}%    │")
        print(f"  │  K-Means            Silhouette      {km.get('silhouette_score',0):6.3f}     │")
        print(f"  │  Hierarchical       Silhouette      {hc.get('silhouette_score',0):6.3f}     │")
        print(f"  │  Isolation Forest   F1-Score        {iso.get('f1_score',0)*100:6.2f}%    │")
        print("  └─────────────────────────────────────────────────┘\n")
        return metrics

    # ─── Step 2: Simulate All Attacks ─────────────────────────────────────────

    def simulate_all_attacks(self, packets_per_attack: int = 80):
        print(f"[ STEP 2 ]  Simulating {len(ATTACK_PROFILES)} attack types "
              f"({packets_per_attack} packets each)...\n")

        for attack_name, profile in ATTACK_PROFILES.items():
            cat      = profile['category']
            desc     = profile['description']
            severity = profile['severity']

            icon = {'DoS':'🔴','Probe':'🟡','R2L':'🟠','U2R':'🟣','normal':'🟢'}.get(cat,'⚪')
            print(f"  {icon}  Simulating: {attack_name.upper():20s}  [{cat}]  {severity.upper()}")

            packets  = generate_attack_wave(attack_name, packets_per_attack)
            detected = 0
            risk_scores  = []
            pred_classes = []
            anomaly_scores = []
            cluster_ids  = []
            t_start = time.perf_counter()

            for pkt in packets:
                result = self.engine.predict(pkt)
                pred_class = result.get('classification', 'unknown')
                risk       = result.get('risk_score', 0)
                is_mal     = result.get('is_malicious', False)
                anomaly    = result.get('anomaly_score', 0)
                cluster    = result.get('cluster_id', -1)

                risk_scores.append(risk)
                pred_classes.append(pred_class)
                anomaly_scores.append(anomaly)
                cluster_ids.append(cluster)

                if cat == 'normal':
                    # For normal, "detected" means correctly classified as normal
                    if not is_mal:
                        detected += 1
                else:
                    if is_mal:
                        detected += 1

                # Accumulate for overall confusion matrix
                self.all_true.append(cat)
                self.all_pred_rf.append(pred_class)

            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000

            detection_rate = detected / packets_per_attack * 100
            avg_risk       = np.mean(risk_scores)
            avg_anomaly    = np.mean(anomaly_scores)

            # Most common predicted class
            from collections import Counter
            pred_dist = dict(Counter(pred_classes))

            self.results[attack_name] = {
                'category':       cat,
                'description':    desc,
                'severity':       severity,
                'packets_sent':   packets_per_attack,
                'detected':       detected,
                'missed':         packets_per_attack - detected,
                'detection_rate': detection_rate,
                'avg_risk_score': avg_risk,
                'avg_anomaly_score': avg_anomaly,
                'pred_distribution': pred_dist,
                'cluster_ids':    cluster_ids,
                'risk_scores':    risk_scores,
                'inference_ms':   elapsed_ms,
                'ms_per_packet':  elapsed_ms / packets_per_attack,
            }

            bar_len = int(detection_rate / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"       Detection: [{bar}] {detection_rate:5.1f}%  "
                  f"Avg Risk: {avg_risk:5.1f}  "
                  f"Speed: {elapsed_ms/packets_per_attack:.2f}ms/pkt")

        print()

    # ─── Step 3: Scenario Tests ────────────────────────────────────────────────

    def simulate_scenarios(self):
        print("[ STEP 3 ]  Running mixed-traffic scenario simulations...\n")

        scenarios = ['realistic', 'heavy_attack', 'stealth', 'multi_vector']
        self.scenario_results = {}

        for scenario in scenarios:
            print(f"  ▶  Scenario: {scenario.upper().replace('_',' ')}")
            packets = generate_mixed_scenario(scenario)
            total = len(packets)

            correct = 0
            malicious_detected = 0
            total_malicious = 0
            risk_scores = []

            for pkt in packets:
                result = self.engine.predict(pkt)
                true_cat = pkt.get('true_category', 'normal')
                pred_class = result.get('classification', 'unknown')
                is_mal = result.get('is_malicious', False)
                risk_scores.append(result.get('risk_score', 0))

                if true_cat != 'normal':
                    total_malicious += 1
                    if is_mal:
                        malicious_detected += 1

                if (true_cat == pred_class) or \
                   (true_cat == 'normal' and not is_mal) or \
                   (true_cat != 'normal' and is_mal):
                    correct += 1

            acc   = correct / total * 100
            dr    = malicious_detected / max(total_malicious, 1) * 100
            fp    = (total - total_malicious - (correct - malicious_detected)) / max(total - total_malicious, 1) * 100
            fp    = max(0, fp)

            self.scenario_results[scenario] = {
                'total_packets':      total,
                'accuracy':           acc,
                'detection_rate':     dr,
                'false_positive_rate': fp,
                'avg_risk':           np.mean(risk_scores),
                'total_malicious':    total_malicious,
                'malicious_detected': malicious_detected,
            }

            print(f"       Packets: {total:4d}  "
                  f"Accuracy: {acc:5.1f}%  "
                  f"Detection Rate: {dr:5.1f}%  "
                  f"FPR: {fp:4.1f}%")

        print()

    # ─── Step 4: Print Full Report ─────────────────────────────────────────────

    def print_report(self):
        print("[ STEP 4 ]  Full Evaluation Report\n")
        print("═" * 90)

        # Per-category summary
        categories = {}
        for name, r in self.results.items():
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'attacks': [], 'rates': [], 'risks': []}
            categories[cat]['attacks'].append(name)
            categories[cat]['rates'].append(r['detection_rate'])
            categories[cat]['risks'].append(r['avg_risk_score'])

        print(f"\n{'ATTACK CATEGORY':<14} {'AVG DETECTION':>14} {'AVG RISK':>10} {'ATTACKS':>8}")
        print("─" * 50)
        for cat, data in categories.items():
            avg_det  = np.mean(data['rates'])
            avg_risk = np.mean(data['risks'])
            icon = {'DoS':'🔴','Probe':'🟡','R2L':'🟠','U2R':'🟣','normal':'🟢'}.get(cat,'⚪')
            print(f"  {icon} {cat:<12} {avg_det:>12.1f}%  {avg_risk:>9.1f}  {len(data['attacks']):>7}")

        print("\n" + "─" * 90)
        print(f"\n{'ATTACK':<20} {'CATEGORY':<10} {'DETECTED':>9} {'MISSED':>7} "
              f"{'RATE':>7} {'AVG RISK':>9} {'MS/PKT':>8}")
        print("─" * 75)
        for name, r in self.results.items():
            rate_col = f"{r['detection_rate']:5.1f}%"
            print(f"  {name:<18} {r['category']:<10} {r['detected']:>8}  "
                  f"{r['missed']:>6}  {rate_col:>7}  "
                  f"{r['avg_risk_score']:>8.1f}  "
                  f"{r['ms_per_packet']:>7.2f}")

        # Overall stats
        all_rates = [r['detection_rate'] for r in self.results.values()
                     if r['category'] != 'normal']
        normal_rates = [r['detection_rate'] for r in self.results.values()
                        if r['category'] == 'normal']

        print("\n" + "═" * 90)
        print(f"\n  Overall Attack Detection Rate : {np.mean(all_rates):6.2f}%")
        print(f"  Normal Traffic Accuracy       : {np.mean(normal_rates):6.2f}%")
        print(f"  Avg Inference Speed           : "
              f"{np.mean([r['ms_per_packet'] for r in self.results.values()]):.3f} ms/packet")

        # Scenario summary
        print("\n" + "─" * 90)
        print(f"\n  {'SCENARIO':<18} {'PACKETS':>8} {'ACCURACY':>10} "
              f"{'DETECTION':>11} {'FPR':>6}")
        print("  " + "─" * 58)
        for sc, r in self.scenario_results.items():
            print(f"  {sc.replace('_',' ').upper():<18} {r['total_packets']:>8}  "
                  f"{r['accuracy']:>8.1f}%  "
                  f"{r['detection_rate']:>9.1f}%  "
                  f"{r['false_positive_rate']:>5.1f}%")
        print()

    # ─── Step 5: Generate All Plots ────────────────────────────────────────────

    def generate_plots(self):
        print("[ STEP 5 ]  Generating evaluation plots...\n")
        self._plot_detection_rates()
        self._plot_risk_distributions()
        self._plot_confusion_matrix()
        self._plot_scenario_comparison()
        self._plot_speed_benchmark()
        self._plot_category_radar()
        print("  ✓ All plots saved to frontend/static/assets/plots/\n")

    def _style_ax(self, ax, title='', xlabel='', ylabel=''):
        ax.set_facecolor(C['surface'])
        ax.tick_params(colors=C['dim'], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C['border'])
        if title:
            ax.set_title(title, color=C['text'], fontsize=10, pad=10,
                         fontfamily='monospace')
        if xlabel:
            ax.set_xlabel(xlabel, color=C['dim'], fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=C['dim'], fontsize=8)
        ax.grid(color=C['border'], linestyle='--', linewidth=0.5, alpha=0.5)

    def _savefig(self, fig, name):
        path = f'frontend/static/assets/plots/{name}'
        fig.savefig(path, dpi=110, bbox_inches='tight',
                    facecolor=C['bg'], edgecolor='none')
        plt.close(fig)
        print(f"  ✓  {name}")

    # ── Plot 1: Detection rates per attack ──
    def _plot_detection_rates(self):
        attacks = list(self.results.keys())
        rates   = [self.results[a]['detection_rate'] for a in attacks]
        cats    = [self.results[a]['category'] for a in attacks]
        colours = [SEV_COLOUR.get(c, C['dim']) for c in cats]

        fig, ax = plt.subplots(figsize=(14, 5), facecolor=C['bg'])
        ax.set_facecolor(C['surface'])
        bars = ax.barh(attacks, rates, color=colours, alpha=0.85,
                       edgecolor=C['border'], linewidth=0.5)

        for bar, rate in zip(bars, rates):
            ax.text(min(rate + 1, 97), bar.get_y() + bar.get_height() / 2,
                    f'{rate:.1f}%', va='center', color=C['text'],
                    fontsize=8, fontfamily='monospace')

        ax.axvline(x=90, color=C['red'], linewidth=1, linestyle='--', alpha=0.6,
                   label='90% threshold')
        ax.set_xlim(0, 105)
        self._style_ax(ax, 'Attack Detection Rate by Attack Type',
                       'Detection Rate (%)', 'Attack Name')
        ax.tick_params(colors=C['text'], labelsize=8)

        legend_patches = [plt.Rectangle((0,0),1,1, color=v, label=k)
                          for k, v in SEV_COLOUR.items()]
        ax.legend(handles=legend_patches, loc='lower right',
                  facecolor=C['surface'], edgecolor=C['border'],
                  labelcolor=C['text'], fontsize=8)
        plt.tight_layout()
        self._savefig(fig, 'sim_detection_rates.png')

    # ── Plot 2: Risk score distributions ──
    def _plot_risk_distributions(self):
        fig, axes = plt.subplots(3, 5, figsize=(18, 10), facecolor=C['bg'])
        axes = axes.flatten()

        for i, (attack_name, r) in enumerate(self.results.items()):
            ax = axes[i]
            colour = SEV_COLOUR.get(r['category'], C['dim'])
            ax.hist(r['risk_scores'], bins=20, color=colour,
                    alpha=0.8, edgecolor=C['border'], linewidth=0.3)
            ax.axvline(x=50, color=C['red'], linewidth=1,
                       linestyle='--', alpha=0.7)
            ax.axvline(x=np.mean(r['risk_scores']), color='white',
                       linewidth=1.2, linestyle='-', alpha=0.8)
            self._style_ax(ax, f"{attack_name}\n[{r['category']}]",
                           'Risk Score', 'Count')
            ax.set_xlim(0, 100)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle('Risk Score Distributions per Attack Type\n'
                     '(red dashed = detection threshold | white = mean)',
                     color=C['text'], fontsize=11, fontfamily='monospace', y=1.01)
        plt.tight_layout()
        self._savefig(fig, 'sim_risk_distributions.png')

    # ── Plot 3: Confusion matrix ──
    def _plot_confusion_matrix(self):
        if not self.all_true or not self.all_pred_rf:
            return

        cats = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
        # Map predictions to categories
        cat_map = {'normal': 'normal', 'DoS': 'DoS', 'Probe': 'Probe',
                   'R2L': 'R2L', 'U2R': 'U2R', 'unknown': 'normal'}
        y_true = self.all_true
        y_pred = [cat_map.get(p, 'normal') for p in self.all_pred_rf]

        cm = confusion_matrix(y_true, y_pred, labels=cats)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C['bg'])

        for ax, data, title, fmt in [
            (axes[0], cm,      'Confusion Matrix (Counts)',       'd'),
            (axes[1], cm_norm, 'Confusion Matrix (Normalised)',   '.2f'),
        ]:
            ax.set_facecolor(C['surface'])
            sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                        xticklabels=cats, yticklabels=cats, ax=ax,
                        linewidths=0.5, linecolor=C['border'],
                        annot_kws={'color': 'white', 'size': 9},
                        cbar_kws={'shrink': 0.8})
            ax.set_title(title, color=C['text'], fontsize=10,
                         fontfamily='monospace', pad=10)
            ax.set_xlabel('Predicted', color=C['dim'], fontsize=9)
            ax.set_ylabel('Actual',    color=C['dim'], fontsize=9)
            ax.tick_params(colors=C['text'], labelsize=9)

        fig.suptitle('ML Model Confusion Matrix — Simulation Results',
                     color=C['text'], fontsize=12, fontfamily='monospace')
        plt.tight_layout()
        self._savefig(fig, 'sim_confusion_matrix.png')

    # ── Plot 4: Scenario comparison ──
    def _plot_scenario_comparison(self):
        scenarios = list(self.scenario_results.keys())
        labels    = [s.replace('_', '\n').upper() for s in scenarios]
        acc       = [self.scenario_results[s]['accuracy']           for s in scenarios]
        dr        = [self.scenario_results[s]['detection_rate']     for s in scenarios]
        fpr       = [self.scenario_results[s]['false_positive_rate'] for s in scenarios]

        x   = np.arange(len(scenarios))
        w   = 0.25
        fig, ax = plt.subplots(figsize=(10, 5), facecolor=C['bg'])
        ax.set_facecolor(C['surface'])

        b1 = ax.bar(x - w, acc, w, label='Accuracy',       color=C['accent'],  alpha=0.85)
        b2 = ax.bar(x,     dr,  w, label='Detection Rate', color=C['green'],   alpha=0.85)
        b3 = ax.bar(x + w, fpr, w, label='False Pos. Rate',color=C['red'],     alpha=0.85)

        for bars in (b1, b2, b3):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', va='bottom',
                        color=C['text'], fontsize=7, fontfamily='monospace')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=C['text'], fontsize=8)
        ax.set_ylim(0, 115)
        self._style_ax(ax, 'Scenario-Level Performance Comparison',
                       'Scenario', 'Score (%)')
        ax.legend(facecolor=C['surface'], edgecolor=C['border'],
                  labelcolor=C['text'], fontsize=9)
        plt.tight_layout()
        self._savefig(fig, 'sim_scenario_comparison.png')

    # ── Plot 5: Speed benchmark ──
    def _plot_speed_benchmark(self):
        attacks  = list(self.results.keys())
        speeds   = [self.results[a]['ms_per_packet'] for a in attacks]
        cats     = [self.results[a]['category'] for a in attacks]
        colours  = [SEV_COLOUR.get(c, C['dim']) for c in cats]

        fig, ax = plt.subplots(figsize=(12, 4), facecolor=C['bg'])
        ax.set_facecolor(C['surface'])
        ax.bar(attacks, speeds, color=colours, alpha=0.85,
               edgecolor=C['border'], linewidth=0.4)
        ax.axhline(y=np.mean(speeds), color=C['accent'], linewidth=1.5,
                   linestyle='--', label=f'Mean: {np.mean(speeds):.2f} ms')
        self._style_ax(ax, 'Inference Speed per Attack Type (ms per packet)',
                       'Attack', 'ms / packet')
        ax.tick_params(axis='x', rotation=45, labelsize=8, colors=C['text'])
        ax.legend(facecolor=C['surface'], edgecolor=C['border'],
                  labelcolor=C['text'], fontsize=9)
        plt.tight_layout()
        self._savefig(fig, 'sim_speed_benchmark.png')

    # ── Plot 6: Radar / spider chart per category ──
    def _plot_category_radar(self):
        cats = ['DoS', 'Probe', 'R2L', 'U2R']
        metrics_labels = ['Detection\nRate', 'Avg Risk\nScore',
                          'Anomaly\nScore', 'Speed\n(inv)']
        N = len(metrics_labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                                 subplot_kw=dict(polar=True),
                                 facecolor=C['bg'])

        for ax, cat in zip(axes, cats):
            attacks_in_cat = [a for a, r in self.results.items()
                              if r['category'] == cat]
            if not attacks_in_cat:
                continue

            det   = np.mean([self.results[a]['detection_rate']    for a in attacks_in_cat]) / 100
            risk  = np.mean([self.results[a]['avg_risk_score']    for a in attacks_in_cat]) / 100
            anom  = np.mean([self.results[a]['avg_anomaly_score'] for a in attacks_in_cat])
            anom  = min(anom, 1.0)
            spd   = 1 - min(np.mean([self.results[a]['ms_per_packet'] for a in attacks_in_cat]) / 10, 1)

            values = [det, risk, anom, spd]
            values += values[:1]

            colour = SEV_COLOUR.get(cat, C['dim'])
            ax.set_facecolor(C['surface'])
            ax.plot(angles, values, color=colour, linewidth=2)
            ax.fill(angles, values, color=colour, alpha=0.2)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_labels, color=C['text'], fontsize=7)
            ax.set_yticklabels([])
            ax.set_ylim(0, 1)
            ax.set_title(cat, color=colour, fontsize=11,
                         fontfamily='monospace', pad=15)
            ax.spines['polar'].set_color(C['border'])
            ax.grid(color=C['border'], linewidth=0.5)

        fig.suptitle('Attack Category Performance Radar',
                     color=C['text'], fontsize=12,
                     fontfamily='monospace', y=1.02)
        plt.tight_layout()
        self._savefig(fig, 'sim_category_radar.png')

    # ─── Step 6: Save JSON results ────────────────────────────────────────────

    def save_results(self):
        output = {
            'generated_at':   __import__('datetime').datetime.utcnow().isoformat(),
            'per_attack':     {
                k: {kk: vv for kk, vv in v.items() if kk != 'risk_scores'}
                for k, v in self.results.items()
            },
            'scenarios':      self.scenario_results,
            'summary': {
                'total_attack_types':    len(self.results),
                'avg_detection_rate':    float(np.mean([
                    r['detection_rate'] for r in self.results.values()
                    if r['category'] != 'normal'])),
                'normal_accuracy':       float(np.mean([
                    r['detection_rate'] for r in self.results.values()
                    if r['category'] == 'normal'])),
                'avg_inference_ms':      float(np.mean([
                    r['ms_per_packet'] for r in self.results.values()])),
            }
        }
        path = 'simulation/results/evaluation_results.json'
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"  ✓ Results saved to {path}\n")
        return output

    # ─── Run Full Pipeline ────────────────────────────────────────────────────

    def run(self, packets_per_attack: int = 80):
        metrics = self.train()
        self.simulate_all_attacks(packets_per_attack)
        self.simulate_scenarios()
        self.print_report()
        self.generate_plots()
        summary = self.save_results()

        print("╔══════════════════════════════════════════════════════╗")
        print("║                 EVALUATION COMPLETE                  ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║  Avg Attack Detection Rate : {summary['summary']['avg_detection_rate']:6.2f}%              ║")
        print(f"║  Normal Traffic Accuracy   : {summary['summary']['normal_accuracy']:6.2f}%              ║")
        print(f"║  Avg Inference Speed       : {summary['summary']['avg_inference_ms']:6.3f} ms/packet       ║")
        print("║  Plots  → frontend/static/assets/plots/sim_*.png    ║")
        print("║  JSON   → simulation/results/evaluation_results.json║")
        print("╚══════════════════════════════════════════════════════╝\n")
        return summary


if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.run(packets_per_attack=100)
