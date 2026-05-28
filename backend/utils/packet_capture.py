import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

_capture_thread: Optional[threading.Thread] = None
_capture_stop_event = threading.Event()
_traffic_buffer: List[Dict[str, Any]] = []
_buffer_lock = threading.Lock()


def _guess_service(port: int) -> str:
    if port in (80, 8080, 443):
        return 'http'
    if port in (22, 2222):
        return 'ssh'
    if port == 53:
        return 'dns'
    if port in (25, 587):
        return 'smtp'
    if port in (110, 143, 993, 995):
        return 'imap'
    return 'other'


def _generate_simulated_packet() -> Dict[str, Any]:
    protocol = random.choice(['tcp', 'udp', 'icmp'])
    src_ip = f'192.168.1.{random.randint(2, 254)}'
    dst_ip = f'10.0.0.{random.randint(2, 254)}'
    dst_port = random.choice([80, 443, 22, 53, 8080, 445, 3389])
    pkt = {
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': random.randint(1024, 65535),
        'dst_port': dst_port,
        'protocol': protocol,
        'packet_size': random.randint(60, 1500),
        'duration': round(random.uniform(0.0, 2.5), 3),
        'src_bytes': random.randint(0, 1500),
        'dst_bytes': random.randint(0, 1500),
        'tcp_flags': random.choice(['SF', 'S0', 'REJ', 'RSTO']),
        'service': _guess_service(dst_port)
    }
    return pkt


def extract_features_from_packet(pkt: Dict[str, Any]) -> Dict[str, Any]:
    src_ip = pkt.get('src_ip', '')
    dst_ip = pkt.get('dst_ip', '')
    protocol = str(pkt.get('protocol', 'tcp')).lower()
    src_port = int(pkt.get('src_port', 0) or 0)
    dst_port = int(pkt.get('dst_port', 0) or 0)
    flags = pkt.get('tcp_flags') or pkt.get('flags') or 'SF'
    service = pkt.get('service') or _guess_service(dst_port)
    features = {
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol_type': protocol,
        'packet_size': int(pkt.get('packet_size', 0) or 0),
        'duration': float(pkt.get('duration', 0.0) or 0.0),
        'flags': flags,
        'tcp_flags': flags,
        'src_bytes': int(pkt.get('src_bytes', 0) or 0),
        'dst_bytes': int(pkt.get('dst_bytes', 0) or 0),
        'service': service,
        'land': 1 if src_ip == dst_ip and src_ip else 0,
        'protocol': protocol
    }
    return features


def _capture_loop(callback=None):
    while not _capture_stop_event.is_set():
        packet = _generate_simulated_packet()
        features = extract_features_from_packet(packet)
        if callback:
            try:
                callback(features, 'simulated')
            except Exception:
                pass
        with _buffer_lock:
            _traffic_buffer.append({'timestamp': datetime.utcnow().isoformat(), **features})
            if len(_traffic_buffer) > 500:
                _traffic_buffer[:] = _traffic_buffer[-500:]
        time.sleep(0.5)


def start_capture(callback=None):
    global _capture_thread
    if is_capturing():
        return
    _capture_stop_event.clear()
    _capture_thread = threading.Thread(target=_capture_loop, args=(callback,), daemon=True)
    _capture_thread.start()


def stop_capture():
    _capture_stop_event.set()
    global _capture_thread
    if _capture_thread is not None:
        _capture_thread.join(timeout=2.0)
        _capture_thread = None


def is_capturing() -> bool:
    return _capture_thread is not None and _capture_thread.is_alive()


def get_traffic_buffer() -> List[Dict[str, Any]]:
    with _buffer_lock:
        return list(_traffic_buffer)


def get_realtime_stats() -> Dict[str, Any]:
    with _buffer_lock:
        total = len(_traffic_buffer)
        malicious = sum(1 for pkt in _traffic_buffer if pkt.get('protocol_type') == 'icmp')
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'packets_captured': total,
        'malicious_packets': malicious,
        'normal_packets': total - malicious,
        'avg_packet_size': round(random.uniform(200, 800), 2),
        'unique_src_ips': len({pkt.get('src_ip') for pkt in _traffic_buffer}),
        'unique_dst_ips': len({pkt.get('dst_ip') for pkt in _traffic_buffer})
    }
