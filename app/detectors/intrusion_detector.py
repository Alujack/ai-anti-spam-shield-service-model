from typing import Dict, List
import re

class IntrusionDetector:
    """Intrusion Detection System"""
    
    def __init__(self):
        self.attack_patterns = {
            'sql_injection': [r"' OR '1'='1", r"'; DROP TABLE", r"UNION SELECT"],
            'xss': [r"<script>", r"javascript:", r"onerror="],
            'command_injection': [r";\s*cat\s+", r"&&\s*ls", r"\|\s*whoami"],
            'path_traversal': [r"\.\./", r"\.\.\\"],
        }
        
    def detect(self, data: Dict) -> Dict:
        """Detect intrusion attempts"""
        
        payload = data.get('payload', '')
        detected_attacks = []
        
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, payload, re.IGNORECASE):
                    detected_attacks.append(attack_type)
                    break
        
        return {
            'is_intrusion': len(detected_attacks) > 0,
            'attack_types': detected_attacks,
            'confidence': 0.9 if detected_attacks else 0.1
        }
    
    def analyze_traffic(self, packets: List[Dict]) -> Dict:
        """Analyze network traffic for anomalies"""
        
        # Placeholder for traffic analysis
        return {
            'anomalies_detected': 0,
            'suspicious_ips': [],
            'attack_types': []
        }
