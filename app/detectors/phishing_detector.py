import re
from typing import Dict, List
import tldextract

class PhishingDetector:
    """Advanced phishing detection module"""
    
    def __init__(self):
        self.suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz']
        self.brand_keywords = ['paypal', 'amazon', 'netflix', 'bank', 'verify']
        
    def detect(self, text: str) -> Dict:
        """Detect phishing indicators in text"""
        
        urls = self.extract_urls(text)
        features = {
            'has_suspicious_urls': self.check_suspicious_urls(urls),
            'has_brand_impersonation': self.check_brand_impersonation(text),
            'has_urgency_language': self.check_urgency_language(text),
            'url_count': len(urls),
            'suspicious_domain_count': sum(1 for url in urls if self.is_suspicious_domain(url))
        }
        
        # Calculate phishing score
        score = 0
        if features['has_suspicious_urls']:
            score += 0.4
        if features['has_brand_impersonation']:
            score += 0.3
        if features['has_urgency_language']:
            score += 0.2
        if features['suspicious_domain_count'] > 0:
            score += 0.1
            
        return {
            'is_phishing': score >= 0.5,
            'confidence': score,
            'features': features
        }
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def is_suspicious_domain(self, url: str) -> bool:
        """Check if domain is suspicious"""
        extracted = tldextract.extract(url)
        return extracted.suffix in self.suspicious_tlds
    
    def check_suspicious_urls(self, urls: List[str]) -> bool:
        """Check for suspicious URLs"""
        return any(self.is_suspicious_domain(url) for url in urls)
    
    def check_brand_impersonation(self, text: str) -> bool:
        """Check for brand impersonation"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.brand_keywords)
    
    def check_urgency_language(self, text: str) -> bool:
        """Check for urgency/panic language"""
        urgency_words = ['urgent', 'immediately', 'verify now', 'account suspended', 
                        'act now', 'limited time', 'expire']
        text_lower = text.lower()
        return any(word in text_lower for word in urgency_words)
