"""
Advanced Threat Detector using Pre-trained Transformers
Replaces weak SMS spam detection with real cybersecurity threat detection
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class AdvancedThreatDetector:
    """
    Production-grade threat detector using pre-trained BERT models
    Detects: Phishing, Malware URLs, Scams, Social Engineering
    """
    
    def __init__(self):
        """Initialize pre-trained models"""
        logger.info("Loading advanced threat detection models...")
        
        # Try to load pre-trained phishing detector
        try:
            self.phishing_detector = pipeline(
                "text-classification",
                model="ealvaradob/bert-finetuned-phishing",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✓ Phishing detector loaded")
        except Exception as e:
            logger.warning(f"Could not load phishing detector: {e}")
            self.phishing_detector = None
        
        # Fallback: General BERT for sentiment/intent
        try:
            self.fallback_detector = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✓ Fallback detector loaded")
        except Exception as e:
            logger.warning(f"Fallback detector failed: {e}")
            self.fallback_detector = None
        
        # Load threat patterns
        self.threat_patterns = self._load_threat_patterns()
        
        logger.info("✓ Advanced Threat Detector initialized")
    
    def _load_threat_patterns(self) -> Dict:
        """Load known threat patterns for rule-based detection"""
        return {
            'phishing_keywords': [
                r'\b(verify|confirm|account|suspended|urgent|immediately|click here|act now)\b',
                r'\b(winner|prize|claim|free|congratulations|won)\b',
                r'\b(password|credit card|social security|bank account|payment)\b',
                r'\b(expired|limited time|offer|discount|deal)\b',
            ],
            'malicious_urls': [
                r'bit\.ly|tinyurl|goo\.gl',  # URL shorteners
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP addresses
                r'\.tk|\.ml|\.ga|\.cf|\.gq',  # Suspicious TLDs
                r'http.*?@',  # @ in URL (obfuscation)
            ],
            'scam_indicators': [
                r'western union|moneygram|wire transfer',
                r'nigerian prince|inheritance|beneficiary',
                r'\$\d+,?\d*,?\d+ (million|thousand)',
                r'cryptocurrency|bitcoin|ethereum.*?(giveaway|investment)',
            ],
            'social_engineering': [
                r'ceo|director|manager.*?(urgent|immediate)',
                r'gift card|itunes|amazon.*?card',
                r'w-2|w2|tax.*?form',
                r'wire.*?transfer.*?(urgent|today)',
            ],
            'malware_indicators': [
                r'\.exe|\.scr|\.bat|\.cmd|\.vbs|\.js',  # Executable extensions
                r'download.*?(now|here|click)',
                r'install.*?(update|patch|fix)',
                r'macro.*?(enable|allow)',
            ]
        }
    
    def detect_threat(self, text: str) -> Dict:
        """
        Main threat detection method
        
        Args:
            text: Text to analyze (email, message, URL, etc.)
            
        Returns:
            Dict with threat analysis results
        """
        if not text or len(text.strip()) < 3:
            return {
                'is_threat': False,
                'threat_type': 'NONE',
                'confidence': 0.0,
                'reason': 'Text too short to analyze'
            }
        
        # Run multiple detection methods
        ml_result = self._ml_detection(text)
        rule_result = self._rule_based_detection(text)
        url_result = self._url_analysis(text)
        feature_result = self._feature_extraction(text)
        
        # Ensemble: Combine all results
        final_result = self._ensemble_decision(
            ml_result, 
            rule_result, 
            url_result, 
            feature_result,
            text
        )
        
        return final_result
    
    def _ml_detection(self, text: str) -> Dict:
        """Machine learning based detection using pre-trained models"""
        results = {}
        
        # Phishing detection
        if self.phishing_detector:
            try:
                phishing_result = self.phishing_detector(text[:512])[0]  # BERT max length
                results['phishing'] = {
                    'label': phishing_result['label'],
                    'score': phishing_result['score']
                }
            except Exception as e:
                logger.error(f"Phishing detection error: {e}")
                results['phishing'] = {'label': 'UNKNOWN', 'score': 0.0}
        
        # Fallback detection
        if self.fallback_detector:
            try:
                fallback_result = self.fallback_detector(text[:512])[0]
                results['sentiment'] = {
                    'label': fallback_result['label'],
                    'score': fallback_result['score']
                }
            except Exception as e:
                logger.error(f"Fallback detection error: {e}")
                results['sentiment'] = {'label': 'UNKNOWN', 'score': 0.0}
        
        return results
    
    def _rule_based_detection(self, text: str) -> Dict:
        """Rule-based detection using threat patterns"""
        text_lower = text.lower()
        matches = {}
        
        for category, patterns in self.threat_patterns.items():
            category_matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    category_matches.append(pattern)
            
            if category_matches:
                matches[category] = {
                    'count': len(category_matches),
                    'patterns': category_matches[:3]  # First 3 matches
                }
        
        return {
            'matches': matches,
            'threat_score': len(matches) * 0.2  # 0.2 per category match
        }
    
    def _url_analysis(self, text: str) -> Dict:
        """Analyze URLs in text"""
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        
        if not urls:
            return {'has_urls': False, 'url_count': 0, 'risk_score': 0.0}
        
        risk_score = 0.0
        suspicious_features = []
        
        for url in urls:
            # Check for IP addresses
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                risk_score += 0.3
                suspicious_features.append('IP address in URL')
            
            # Check for suspicious TLDs
            if re.search(r'\.(tk|ml|ga|cf|gq)', url, re.IGNORECASE):
                risk_score += 0.4
                suspicious_features.append('Suspicious TLD')
            
            # Check for @ symbol (obfuscation)
            if '@' in url:
                risk_score += 0.5
                suspicious_features.append('@ symbol in URL')
            
            # Check for excessive subdomains
            if url.count('.') > 4:
                risk_score += 0.2
                suspicious_features.append('Excessive subdomains')
            
            # Check for URL shorteners
            if re.search(r'(bit\.ly|tinyurl|goo\.gl|ow\.ly)', url, re.IGNORECASE):
                risk_score += 0.3
                suspicious_features.append('URL shortener')
        
        return {
            'has_urls': True,
            'url_count': len(urls),
            'urls': urls[:5],  # First 5 URLs
            'risk_score': min(risk_score, 1.0),
            'suspicious_features': suspicious_features
        }
    
    def _feature_extraction(self, text: str) -> Dict:
        """Extract statistical features from text"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'currency_count': len(re.findall(r'[$£€¥]', text)),
            'email_count': len(re.findall(r'\S+@\S+', text)),
            'phone_count': len(re.findall(r'\b\d{10,}\b', text)),
        }
        
        # Calculate suspicion score based on features
        suspicion_score = 0.0
        
        if features['uppercase_ratio'] > 0.3:
            suspicion_score += 0.2
        if features['exclamation_count'] > 3:
            suspicion_score += 0.2
        if features['currency_count'] > 0:
            suspicion_score += 0.1
        if features['special_char_ratio'] > 0.2:
            suspicion_score += 0.1
        
        features['suspicion_score'] = min(suspicion_score, 1.0)
        
        return features
    
    def _ensemble_decision(self, ml_result: Dict, rule_result: Dict, 
                          url_result: Dict, feature_result: Dict, text: str) -> Dict:
        """
        Ensemble decision combining all detection methods
        """
        total_score = 0.0
        threat_indicators = []
        threat_type = 'UNKNOWN'
        
        # Weight 1: ML Detection (40%)
        if ml_result.get('phishing'):
            phishing_score = ml_result['phishing']['score']
            if ml_result['phishing']['label'] in ['PHISHING', 'LABEL_1', '1']:
                total_score += phishing_score * 0.4
                threat_indicators.append(f"ML Phishing Score: {phishing_score:.2f}")
                threat_type = 'PHISHING'
        
        # Weight 2: Rule-based Detection (30%)
        rule_score = rule_result.get('threat_score', 0.0)
        total_score += rule_score * 0.3
        
        if rule_result.get('matches'):
            for category, data in rule_result['matches'].items():
                threat_indicators.append(f"{category}: {data['count']} matches")
                
                # Determine threat type
                if 'phishing' in category and threat_type == 'UNKNOWN':
                    threat_type = 'PHISHING'
                elif 'scam' in category and threat_type == 'UNKNOWN':
                    threat_type = 'SCAM'
                elif 'malware' in category and threat_type == 'UNKNOWN':
                    threat_type = 'MALWARE'
                elif 'social_engineering' in category and threat_type == 'UNKNOWN':
                    threat_type = 'SOCIAL_ENGINEERING'
        
        # Weight 3: URL Analysis (20%)
        url_score = url_result.get('risk_score', 0.0)
        total_score += url_score * 0.2
        
        if url_result.get('suspicious_features'):
            threat_indicators.extend(url_result['suspicious_features'])
        
        # Weight 4: Feature Analysis (10%)
        feature_score = feature_result.get('suspicion_score', 0.0)
        total_score += feature_score * 0.1
        
        # Normalize total score
        total_score = min(total_score, 1.0)
        
        # Decision threshold
        is_threat = total_score >= 0.5
        
        # Determine confidence
        if total_score >= 0.8:
            confidence_level = 'VERY HIGH'
        elif total_score >= 0.6:
            confidence_level = 'HIGH'
        elif total_score >= 0.4:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'LOW'
        
        return {
            'is_threat': is_threat,
            'threat_type': threat_type if is_threat else 'NONE',
            'confidence': total_score,
            'confidence_level': confidence_level,
            'threat_indicators': threat_indicators,
            'details': {
                'ml_detection': ml_result,
                'rule_based': rule_result,
                'url_analysis': url_result,
                'features': feature_result
            },
            'recommendation': self._get_recommendation(is_threat, threat_type, total_score)
        }
    
    def _get_recommendation(self, is_threat: bool, threat_type: str, confidence: float) -> str:
        """Get action recommendation based on threat level"""
        if not is_threat:
            return "Message appears safe. No action needed."
        
        if confidence >= 0.8:
            return f"CRITICAL: High-confidence {threat_type} detected. Block immediately and report."
        elif confidence >= 0.6:
            return f"WARNING: Likely {threat_type}. Exercise caution and verify sender."
        else:
            return f"SUSPICIOUS: Possible {threat_type}. Review carefully before taking action."
    
    def batch_detect(self, texts: List[str]) -> List[Dict]:
        """Detect threats in multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.detect_threat(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch detection error: {e}")
                results.append({
                    'is_threat': False,
                    'threat_type': 'ERROR',
                    'confidence': 0.0,
                    'error': str(e)
                })
        return results


# Singleton instance
_detector_instance = None

def get_detector() -> AdvancedThreatDetector:
    """Get or create detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AdvancedThreatDetector()
    return _detector_instance


if __name__ == '__main__':
    # Test the detector
    detector = AdvancedThreatDetector()
    
    test_cases = [
        "Congratulations! You've won $1,000,000. Click here to claim: http://bit.ly/fake123",
        "Hey, are we still meeting for lunch tomorrow at noon?",
        "URGENT: Your account has been suspended. Verify immediately at http://192.168.1.1/verify",
        "Free iPhone 15 Pro Max! Limited time offer. Click now: http://malicious.tk/offer",
        "Hi team, please find the quarterly report attached. Let me know if you have questions.",
        "Investment opportunity! Send Bitcoin to get 10x returns guaranteed!",
        "Your package delivery failed. Download this file to reschedule: suspicious.exe"
    ]
    
    print("🔍 Testing Advanced Threat Detector\n")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        result = detector.detect_threat(text)
        
        print(f"\nTest {i}:")
        print(f"Text: {text[:80]}...")
        print(f"Threat: {result['is_threat']}")
        print(f"Type: {result['threat_type']}")
        print(f"Confidence: {result['confidence']:.2%} ({result['confidence_level']})")
        print(f"Indicators: {', '.join(result['threat_indicators'][:3])}")
        print(f"Action: {result['recommendation']}")
        print("-" * 80)

