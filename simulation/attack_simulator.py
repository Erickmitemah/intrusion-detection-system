"""
NetGuard IDS — Attack Simulator
Generates realistic synthetic network attack traffic for all 5 KDD-99 categories.
Used to test ML model detection accuracy in real-time simulation.
"""

import numpy as np
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple


# ─── Attack Profile Definitions ────────────────────────────────────────────────

ATTACK_PROFILES = {

    # ── DoS Attacks ──────────────────────────────────────────────────────────
    'neptune': {
        'category': 'DoS',
        'description': 'SYN flood — sends half-open TCP connections exhausting server resources',
        'severity': 'critical',
        'features': {
            'duration': 0, 'protocol_type': 'tcp', 'service': 'http',
            'flag': 'S0',  # SYN sent, no reply
            'src_bytes': 0, 'dst_bytes': 0, 'land': 0,
            'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
            'num_failed_logins': 0, 'logged_in': 0,
            'count': lambda: random.randint(450, 511),
            'srv_count': lambda: random.randint(450, 511),
            'serror_rate': 1.0, 'srv_serror_rate': 1.0,
            'rerror_rate': 0.0, 'same_srv_rate': 1.0,
            'diff_srv_rate': 0.0, 'dst_host_count': 255,
            'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 1.0,
            'dst_host_serror_rate': 1.0,
        }
    },

    'smurf': {
        'category': 'DoS',
        'description': 'ICMP amplification — spoofed broadcast ping flood',
        'severity': 'critical',
        'features': {
            'duration': 0, 'protocol_type': 'icmp', 'service': 'eco_i',
            'flag': 'SF', 'src_bytes': 1032, 'dst_bytes': 0,
            'land': 0, 'wrong_fragment': 0, 'urgent': 0,
            'count': lambda: random.randint(480, 511),
            'srv_count': lambda: random.randint(480, 511),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': 1.0, 'diff_srv_rate': 0.0,
            'dst_host_count': 255, 'dst_host_srv_count': 255,
            'dst_host_same_srv_rate': 1.0,
        }
    },

    'teardrop': {
        'category': 'DoS',
        'description': 'Fragmented packet attack — overlapping IP fragments crash kernel',
        'severity': 'high',
        'features': {
            'duration': 0, 'protocol_type': 'udp', 'service': 'private',
            'flag': 'SF',
            'src_bytes': lambda: random.randint(28, 100),
            'dst_bytes': 0,
            'wrong_fragment': lambda: random.randint(1, 3),
            'urgent': 0, 'land': 0,
            'count': lambda: random.randint(1, 10),
            'serror_rate': 0.0, 'rerror_rate': 0.06,
            'same_srv_rate': 1.0,
        }
    },

    'back': {
        'category': 'DoS',
        'description': 'Apache DoS — sends malformed HTTP requests to overwhelm web server',
        'severity': 'high',
        'features': {
            'duration': lambda: random.uniform(0.1, 2.0),
            'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
            'src_bytes': lambda: random.randint(2000, 54540),
            'dst_bytes': lambda: random.randint(1, 5),
            'logged_in': 1, 'hot': lambda: random.randint(1, 10),
            'count': lambda: random.randint(1, 5),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': lambda: random.uniform(0.03, 0.06),
        }
    },

    # ── Probe Attacks ─────────────────────────────────────────────────────────
    'portsweep': {
        'category': 'Probe',
        'description': 'Port sweep — scans a single host for open services across many ports',
        'severity': 'medium',
        'features': {
            'duration': lambda: random.uniform(0.0, 1.0),
            'protocol_type': 'tcp', 'service': 'private', 'flag': 'REJ',
            'src_bytes': lambda: random.randint(0, 10),
            'dst_bytes': 0, 'land': 0,
            'count': lambda: random.randint(1, 15),
            'srv_count': lambda: random.randint(1, 10),
            'serror_rate': 0.0,
            'rerror_rate': lambda: random.uniform(0.6, 1.0),
            'same_srv_rate': lambda: random.uniform(0.0, 0.05),
            'diff_srv_rate': lambda: random.uniform(0.9, 1.0),
            'dst_host_count': lambda: random.randint(1, 30),
            'dst_host_same_srv_rate': lambda: random.uniform(0.0, 0.1),
            'dst_host_diff_srv_rate': lambda: random.uniform(0.5, 1.0),
        }
    },

    'nmap': {
        'category': 'Probe',
        'description': 'Nmap scan — network mapper fingerprinting OS and services',
        'severity': 'medium',
        'features': {
            'duration': 0, 'protocol_type': 'tcp', 'service': 'private',
            'flag': 'S0',
            'src_bytes': 0, 'dst_bytes': 0, 'land': 0,
            'count': lambda: random.randint(1, 20),
            'srv_count': lambda: random.randint(1, 5),
            'serror_rate': lambda: random.uniform(0.0, 1.0),
            'rerror_rate': lambda: random.uniform(0.0, 1.0),
            'same_srv_rate': lambda: random.uniform(0.0, 0.5),
            'diff_srv_rate': lambda: random.uniform(0.5, 1.0),
            'dst_host_count': lambda: random.randint(10, 255),
            'dst_host_same_srv_rate': lambda: random.uniform(0.0, 0.5),
        }
    },

    'ipsweep': {
        'category': 'Probe',
        'description': 'IP sweep — pings a range of IPs to find live hosts (network discovery)',
        'severity': 'low',
        'features': {
            'duration': lambda: random.uniform(0.0, 0.3),
            'protocol_type': 'icmp', 'service': 'eco_i', 'flag': 'SF',
            'src_bytes': lambda: random.randint(18, 28),
            'dst_bytes': lambda: random.randint(18, 28),
            'count': lambda: random.randint(1, 511),
            'srv_count': lambda: random.randint(1, 30),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': lambda: random.uniform(0.0, 0.5),
            'diff_srv_rate': lambda: random.uniform(0.0, 1.0),
            'dst_host_count': lambda: random.randint(200, 255),
            'dst_host_srv_diff_host_rate': lambda: random.uniform(0.5, 1.0),
        }
    },

    # ── R2L Attacks ───────────────────────────────────────────────────────────
    'guess_passwd': {
        'category': 'R2L',
        'description': 'Brute force login — repeatedly tries passwords to gain system access',
        'severity': 'high',
        'features': {
            'duration': lambda: random.uniform(0.0, 5.0),
            'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF',
            'src_bytes': lambda: random.randint(80, 200),
            'dst_bytes': lambda: random.randint(80, 200),
            'logged_in': 0,
            'num_failed_logins': lambda: random.randint(5, 20),
            'count': lambda: random.randint(10, 50),
            'srv_count': lambda: random.randint(10, 50),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': lambda: random.uniform(0.5, 1.0),
        }
    },

    'ftp_write': {
        'category': 'R2L',
        'description': 'FTP write exploit — uploads malicious files via anonymous FTP write access',
        'severity': 'high',
        'features': {
            'duration': lambda: random.uniform(1.0, 10.0),
            'protocol_type': 'tcp', 'service': 'ftp_data', 'flag': 'SF',
            'src_bytes': lambda: random.randint(100, 9000),
            'dst_bytes': lambda: random.randint(100, 4000),
            'logged_in': 1, 'num_file_creations': lambda: random.randint(1, 5),
            'count': lambda: random.randint(1, 10),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': lambda: random.uniform(0.8, 1.0),
        }
    },

    'imap': {
        'category': 'R2L',
        'description': 'IMAP exploit — remote buffer overflow in email server to gain shell',
        'severity': 'high',
        'features': {
            'duration': lambda: random.uniform(0.0, 2.0),
            'protocol_type': 'tcp', 'service': 'imap4', 'flag': 'SF',
            'src_bytes': lambda: random.randint(100, 1000),
            'dst_bytes': lambda: random.randint(100, 1000),
            'logged_in': lambda: random.choice([0, 1]),
            'num_failed_logins': lambda: random.randint(0, 3),
            'count': lambda: random.randint(1, 5),
            'serror_rate': 0.0,
        }
    },

    # ── U2R Attacks ───────────────────────────────────────────────────────────
    'buffer_overflow': {
        'category': 'U2R',
        'description': 'Buffer overflow — overwrites memory to execute arbitrary code as root',
        'severity': 'critical',
        'features': {
            'duration': lambda: random.uniform(1.0, 20.0),
            'protocol_type': 'tcp', 'service': 'telnet', 'flag': 'SF',
            'src_bytes': lambda: random.randint(200, 3000),
            'dst_bytes': lambda: random.randint(100, 1500),
            'logged_in': 1,
            'root_shell': 1,
            'num_root': lambda: random.randint(1, 10),
            'num_shells': lambda: random.randint(1, 3),
            'hot': lambda: random.randint(5, 30),
            'count': lambda: random.randint(1, 5),
            'serror_rate': 0.0,
        }
    },

    'rootkit': {
        'category': 'U2R',
        'description': 'Rootkit install — installs persistent backdoor with root privileges',
        'severity': 'critical',
        'features': {
            'duration': lambda: random.uniform(5.0, 60.0),
            'protocol_type': 'tcp', 'service': 'telnet', 'flag': 'SF',
            'src_bytes': lambda: random.randint(500, 5000),
            'dst_bytes': lambda: random.randint(200, 3000),
            'logged_in': 1,
            'root_shell': 1,
            'su_attempted': 1,
            'num_root': lambda: random.randint(2, 20),
            'num_file_creations': lambda: random.randint(1, 10),
            'num_access_files': lambda: random.randint(1, 5),
            'hot': lambda: random.randint(10, 50),
            'count': lambda: random.randint(1, 3),
        }
    },

    'loadmodule': {
        'category': 'U2R',
        'description': 'Kernel module load — inserts malicious kernel module for full system control',
        'severity': 'critical',
        'features': {
            'duration': lambda: random.uniform(0.5, 5.0),
            'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF',
            'src_bytes': lambda: random.randint(100, 2000),
            'dst_bytes': lambda: random.randint(100, 1000),
            'logged_in': 1, 'root_shell': 1,
            'num_root': lambda: random.randint(1, 5),
            'su_attempted': 1,
            'hot': lambda: random.randint(3, 20),
            'count': lambda: random.randint(1, 5),
        }
    },

    # ── Normal Traffic ────────────────────────────────────────────────────────
    'normal_http': {
        'category': 'normal',
        'description': 'Normal HTTP web browsing traffic',
        'severity': 'none',
        'features': {
            'duration': lambda: random.uniform(0.1, 10.0),
            'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
            'src_bytes': lambda: random.randint(200, 5000),
            'dst_bytes': lambda: random.randint(500, 50000),
            'logged_in': 1, 'hot': 0,
            'count': lambda: random.randint(1, 20),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': lambda: random.uniform(0.8, 1.0),
        }
    },

    'normal_ftp': {
        'category': 'normal',
        'description': 'Normal FTP file transfer session',
        'severity': 'none',
        'features': {
            'duration': lambda: random.uniform(1.0, 30.0),
            'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF',
            'src_bytes': lambda: random.randint(300, 8000),
            'dst_bytes': lambda: random.randint(300, 8000),
            'logged_in': 1, 'count': lambda: random.randint(1, 10),
            'serror_rate': 0.0, 'rerror_rate': 0.0,
            'same_srv_rate': lambda: random.uniform(0.9, 1.0),
        }
    },
}


# ─── Feature Defaults ─────────────────────────────────────────────────────────

FEATURE_DEFAULTS = {
    'duration': 0, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'SF',
    'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
    'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
    'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0,
    'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
    'num_access_files': 0, 'num_outbound_cmds': 0,
    'is_host_login': 0, 'is_guest_login': 0,
    'count': 1, 'srv_count': 1, 'serror_rate': 0.0,
    'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
    'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
    'dst_host_count': 1, 'dst_host_srv_count': 1,
    'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
    'dst_host_same_src_port_rate': 0.0, 'dst_host_srv_diff_host_rate': 0.0,
    'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
    'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0,
}


def _resolve(val):
    """Resolve a feature value — call if callable, else return directly."""
    return val() if callable(val) else val


def generate_attack_packet(attack_name: str, attack_num: int = 1) -> Dict:
    """Generate a single realistic feature vector for a given attack type."""
    if attack_name not in ATTACK_PROFILES:
        raise ValueError(f"Unknown attack: {attack_name}. Choose from: {list(ATTACK_PROFILES.keys())}")

    profile = ATTACK_PROFILES[attack_name]
    features = dict(FEATURE_DEFAULTS)

    # Apply attack-specific overrides
    for k, v in profile['features'].items():
        features[k] = _resolve(v)

    # Add metadata
    ip_pool = [f"192.168.{random.randint(1,5)}.{random.randint(1,254)}" for _ in range(3)]
    ip_pool += [f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}" for _ in range(2)]

    features['src_ip'] = random.choice(ip_pool)
    features['dst_ip'] = f"10.0.0.{random.randint(1, 20)}"
    features['src_port'] = random.randint(1024, 65535)
    features['dst_port'] = {
        'http': 80, 'ftp': 21, 'ftp_data': 20, 'telnet': 23,
        'smtp': 25, 'ssh': 22, 'imap4': 143, 'eco_i': 7, 'private': random.randint(1024, 65535)
    }.get(features.get('service', 'http'), 80)
    features['packet_size'] = features['src_bytes'] + features['dst_bytes']
    features['timestamp'] = datetime.utcnow().isoformat()
    features['attack_name'] = attack_name
    features['true_category'] = profile['category']
    features['attack_num'] = attack_num

    return features


def generate_attack_wave(attack_name: str, n_packets: int = 50) -> List[Dict]:
    """Generate a wave of packets for a single attack scenario."""
    return [generate_attack_packet(attack_name, i + 1) for i in range(n_packets)]


def generate_mixed_scenario(scenario: str = 'realistic') -> List[Dict]:
    """
    Generate a full mixed-traffic simulation scenario.
    scenarios: 'realistic', 'heavy_attack', 'stealth', 'multi_vector'
    """
    scenarios = {
        'realistic': [
            ('normal_http', 60), ('normal_ftp', 20),
            ('neptune', 10), ('portsweep', 5), ('guess_passwd', 3),
            ('buffer_overflow', 2),
        ],
        'heavy_attack': [
            ('neptune', 40), ('smurf', 30), ('teardrop', 15),
            ('portsweep', 10), ('nmap', 5),
        ],
        'stealth': [
            ('normal_http', 70), ('normal_ftp', 10),
            ('ipsweep', 8), ('guess_passwd', 7),
            ('rootkit', 3), ('loadmodule', 2),
        ],
        'multi_vector': [
            ('neptune', 20), ('smurf', 10), ('portsweep', 15),
            ('nmap', 10), ('guess_passwd', 15), ('ftp_write', 5),
            ('buffer_overflow', 10), ('rootkit', 5), ('normal_http', 10),
        ],
    }

    mix = scenarios.get(scenario, scenarios['realistic'])
    packets = []
    for attack_name, count in mix:
        packets.extend(generate_attack_wave(attack_name, count))

    random.shuffle(packets)
    return packets


def get_all_attack_names() -> List[str]:
    return list(ATTACK_PROFILES.keys())


def get_profile(attack_name: str) -> Dict:
    return ATTACK_PROFILES.get(attack_name, {})
