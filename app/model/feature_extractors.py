"""
Feature Extractors for Phishing Detection
Extracts features from URLs and text messages for ML-based phishing detection

Features are designed based on research on phishing characteristics:
- URL structure and patterns
- Text linguistic patterns
- Brand impersonation indicators
- Urgency and threat language
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs
from collections import Counter

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False
    logging.warning("tldextract not installed, some URL features will be limited")

logger = logging.getLogger(__name__)


class URLFeatureExtractor:
    """
    Extract features from URLs for phishing detection

    Features based on:
    - URL structure (length, path depth, query params)
    - Domain characteristics (TLD, subdomains, IP addresses)
    - Suspicious patterns (typosquatting, brand impersonation)
    - Security indicators (HTTPS, obfuscation)
    """

    # Known brands commonly impersonated in phishing
    KNOWN_BRANDS = [
        'paypal', 'amazon', 'netflix', 'apple', 'microsoft', 'google',
        'facebook', 'instagram', 'whatsapp', 'chase', 'wellsfargo',
        'bankofamerica', 'citibank', 'usps', 'fedex', 'ups', 'dhl',
        'irs', 'ssa', 'walmart', 'target', 'ebay', 'linkedin',
        'dropbox', 'adobe', 'spotify', 'yahoo', 'outlook', 'office365',
        'docusign', 'zoom', 'slack', 'github', 'twitter'
    ]

    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = [
        'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click',
        'link', 'info', 'online', 'site', 'website', 'space', 'pw',
        'cc', 'su', 'buzz', 'fit', 'loan', 'download', 'stream'
    ]

    # URL shorteners commonly abused
    URL_SHORTENERS = [
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'j.mp', 'tr.im', 'cli.gs', 'short.to',
        'budurl.com', 'clck.ru', 'cutt.ly', 'rb.gy', 'shorturl.at'
    ]

    # Suspicious keywords in URLs
    SUSPICIOUS_URL_KEYWORDS = [
        'login', 'signin', 'verify', 'secure', 'account', 'update',
        'confirm', 'banking', 'password', 'credential', 'authenticate',
        'validation', 'suspend', 'locked', 'unusual', 'activity'
    ]

    def __init__(self):
        """Initialize the URL feature extractor"""
        self.brand_pattern = re.compile(
            r'\b(' + '|'.join(self.KNOWN_BRANDS) + r')\b',
            re.IGNORECASE
        )

    def extract(self, url: str) -> Dict[str, float]:
        """
        Extract all features from a URL

        Args:
            url: URL string to analyze

        Returns:
            Dictionary of feature names to values
        """
        if not url:
            return self._empty_features()

        url = url.strip()

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception:
            return self._empty_features()

        # Extract TLD info
        if HAS_TLDEXTRACT:
            extracted = tldextract.extract(url)
            domain = extracted.domain
            suffix = extracted.suffix
            subdomain = extracted.subdomain
        else:
            domain = parsed.netloc
            suffix = ''
            subdomain = ''

        features = {}

        # Length features
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed.netloc)
        features['path_length'] = len(parsed.path)

        # Structure features
        features['subdomain_count'] = len(subdomain.split('.')) if subdomain else 0
        features['path_depth'] = len([p for p in parsed.path.split('/') if p])
        features['query_param_count'] = len(parse_qs(parsed.query))

        # Character features
        features['digit_count'] = sum(c.isdigit() for c in url)
        features['special_char_count'] = sum(not c.isalnum() and c not in ':/.' for c in url)
        features['digit_ratio'] = features['digit_count'] / max(len(url), 1)

        # Security features
        features['has_https'] = 1.0 if parsed.scheme == 'https' else 0.0
        features['has_http'] = 1.0 if parsed.scheme == 'http' else 0.0

        # Suspicious patterns
        features['has_ip_address'] = 1.0 if self._has_ip_address(parsed.netloc) else 0.0
        features['has_at_symbol'] = 1.0 if '@' in url else 0.0
        features['has_double_slash'] = 1.0 if '//' in parsed.path else 0.0
        features['has_hexadecimal'] = 1.0 if self._has_hex_encoding(url) else 0.0

        # TLD analysis
        features['suspicious_tld'] = 1.0 if suffix.lower() in self.SUSPICIOUS_TLDS else 0.0
        features['tld_length'] = len(suffix)

        # Brand and keyword detection
        features['has_brand_name'] = 1.0 if self.brand_pattern.search(url) else 0.0
        features['suspicious_keyword_count'] = self._count_suspicious_keywords(url)

        # URL shortener detection
        features['is_shortened'] = 1.0 if self._is_shortened_url(url) else 0.0

        # Entropy (randomness)
        features['domain_entropy'] = self._calculate_entropy(domain)
        features['url_entropy'] = self._calculate_entropy(url)

        # Typosquatting indicators
        features['has_hyphen_in_domain'] = 1.0 if '-' in domain else 0.0
        features['consecutive_digits'] = self._max_consecutive_digits(domain)

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary"""
        return {
            'url_length': 0, 'domain_length': 0, 'path_length': 0,
            'subdomain_count': 0, 'path_depth': 0, 'query_param_count': 0,
            'digit_count': 0, 'special_char_count': 0, 'digit_ratio': 0,
            'has_https': 0, 'has_http': 0, 'has_ip_address': 0,
            'has_at_symbol': 0, 'has_double_slash': 0, 'has_hexadecimal': 0,
            'suspicious_tld': 0, 'tld_length': 0, 'has_brand_name': 0,
            'suspicious_keyword_count': 0, 'is_shortened': 0,
            'domain_entropy': 0, 'url_entropy': 0, 'has_hyphen_in_domain': 0,
            'consecutive_digits': 0
        }

    def _has_ip_address(self, netloc: str) -> bool:
        """Check if netloc contains an IP address"""
        # IPv4 pattern
        ipv4_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^\[?[0-9a-fA-F:]+\]?'

        host = netloc.split(':')[0]  # Remove port
        return bool(re.match(ipv4_pattern, host) or re.match(ipv6_pattern, host))

    def _has_hex_encoding(self, url: str) -> bool:
        """Check for hexadecimal encoding in URL"""
        return bool(re.search(r'%[0-9a-fA-F]{2}', url))

    def _count_suspicious_keywords(self, url: str) -> int:
        """Count suspicious keywords in URL"""
        url_lower = url.lower()
        return sum(1 for keyword in self.SUSPICIOUS_URL_KEYWORDS if keyword in url_lower)

    def _is_shortened_url(self, url: str) -> bool:
        """Check if URL is from a shortening service"""
        url_lower = url.lower()
        return any(shortener in url_lower for shortener in self.URL_SHORTENERS)

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0

        freq = Counter(text.lower())
        length = len(text)
        entropy = 0.0

        for count in freq.values():
            if count > 0:
                prob = count / length
                entropy -= prob * math.log2(prob)

        return entropy

    def _max_consecutive_digits(self, text: str) -> int:
        """Find maximum consecutive digits in text"""
        max_count = 0
        current_count = 0

        for char in text:
            if char.isdigit():
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return list(self._empty_features().keys())


class TextFeatureExtractor:
    """
    Extract features from text messages for phishing detection

    Features based on:
    - Linguistic patterns (urgency, threats)
    - Content analysis (financial, credential requests)
    - Structure (URLs, contact info)
    - Style (capitalization, punctuation)
    """

    # Urgency patterns
    URGENCY_PATTERNS = [
        r'\b(urgent|immediately|asap|now|today|expires?|limited time)\b',
        r'\b(act now|hurry|don\'t wait|final notice|last chance)\b',
        r'\b(within \d+ (hours?|days?|minutes?))\b',
        r'\b(deadline|expire|suspend|terminate)\b',
    ]

    # Threat patterns
    THREAT_PATTERNS = [
        r'\b(account (suspended|locked|compromised|restricted|disabled))\b',
        r'\b(unauthorized (access|activity|transaction))\b',
        r'\b(security (alert|warning|breach|issue))\b',
        r'\b(suspicious (activity|login|attempt))\b',
        r'\b(will be (closed|terminated|deleted|suspended))\b',
    ]

    # Credential request patterns
    CREDENTIAL_PATTERNS = [
        r'\b(verify (your )?(account|identity|information|password))\b',
        r'\b(confirm (your )?(account|identity|details|password))\b',
        r'\b(update (your )?(account|information|details|password))\b',
        r'\b(enter (your )?(password|pin|ssn|credit card))\b',
        r'\b(provide (your )?(password|credentials|information))\b',
    ]

    # Financial patterns
    FINANCIAL_PATTERNS = [
        r'\b(bank|credit card|debit card|payment|transaction)\b',
        r'\b(wire transfer|bitcoin|cryptocurrency|investment)\b',
        r'\b(refund|reimbursement|compensation|prize|winner)\b',
        r'[$£€¥]\s?\d+[,\d]*',
        r'\b\d+\s?(dollars?|pounds?|euros?|usd|gbp|eur)\b',
    ]

    # Action request patterns
    ACTION_PATTERNS = [
        r'\b(click (here|the link|below|this))\b',
        r'\b(follow (this|the) link)\b',
        r'\b(visit (this|our) (website|page|link))\b',
        r'\b(download (the )?(attachment|file|document))\b',
        r'\b(open (the )?(attachment|file|document))\b',
    ]

    # Impersonation patterns
    IMPERSONATION_PATTERNS = [
        r'\b(dear (customer|user|member|client|valued))\b',
        r'\b(from (the )?(support|team|department|administration))\b',
        r'\b(official (notice|notification|communication))\b',
        r'\b(customer (service|support|care))\b',
    ]

    def __init__(self):
        """Initialize the text feature extractor"""
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            re.IGNORECASE
        )
        self.phone_pattern = re.compile(
            r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        )

        # Compile patterns
        self.urgency_patterns = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS]
        self.threat_patterns = [re.compile(p, re.IGNORECASE) for p in self.THREAT_PATTERNS]
        self.credential_patterns = [re.compile(p, re.IGNORECASE) for p in self.CREDENTIAL_PATTERNS]
        self.financial_patterns = [re.compile(p, re.IGNORECASE) for p in self.FINANCIAL_PATTERNS]
        self.action_patterns = [re.compile(p, re.IGNORECASE) for p in self.ACTION_PATTERNS]
        self.impersonation_patterns = [re.compile(p, re.IGNORECASE) for p in self.IMPERSONATION_PATTERNS]

    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text

        Args:
            text: Text string to analyze

        Returns:
            Dictionary of feature names to values
        """
        if not text:
            return self._empty_features()

        text = text.strip()
        text_lower = text.lower()
        words = text.split()
        word_count = len(words)

        features = {}

        # Length features
        features['text_length'] = len(text)
        features['word_count'] = word_count

        # Content detection
        urls = self.url_pattern.findall(text)
        emails = self.email_pattern.findall(text)
        phones = self.phone_pattern.findall(text)

        features['url_count'] = len(urls)
        features['email_count'] = len(emails)
        features['phone_count'] = len(phones)
        features['has_url'] = 1.0 if urls else 0.0
        features['has_email'] = 1.0 if emails else 0.0
        features['has_phone'] = 1.0 if phones else 0.0

        # Pattern matching scores
        features['urgency_score'] = self._count_pattern_matches(text, self.urgency_patterns)
        features['threat_score'] = self._count_pattern_matches(text, self.threat_patterns)
        features['credential_score'] = self._count_pattern_matches(text, self.credential_patterns)
        features['financial_score'] = self._count_pattern_matches(text, self.financial_patterns)
        features['action_score'] = self._count_pattern_matches(text, self.action_patterns)
        features['impersonation_score'] = self._count_pattern_matches(text, self.impersonation_patterns)

        # Style features
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['punctuation_ratio'] = sum(1 for c in text if c in '.,!?;:') / max(len(text), 1)

        # Word-level features
        features['avg_word_length'] = sum(len(w) for w in words) / max(word_count, 1)
        features['short_word_ratio'] = sum(1 for w in words if len(w) <= 3) / max(word_count, 1)
        features['long_word_ratio'] = sum(1 for w in words if len(w) >= 10) / max(word_count, 1)

        # Capitalization patterns
        features['all_caps_word_count'] = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['capitalized_word_ratio'] = sum(1 for w in words if w and w[0].isupper()) / max(word_count, 1)

        # Special content
        features['has_currency'] = 1.0 if re.search(r'[$£€¥]', text) else 0.0
        features['has_percentage'] = 1.0 if re.search(r'\d+%', text) else 0.0
        features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / max(len(text), 1)

        # Combined score (weighted sum of pattern scores)
        features['phishing_indicator_score'] = (
            features['urgency_score'] * 0.2 +
            features['threat_score'] * 0.25 +
            features['credential_score'] * 0.25 +
            features['action_score'] * 0.15 +
            features['impersonation_score'] * 0.15
        )

        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dictionary"""
        return {
            'text_length': 0, 'word_count': 0,
            'url_count': 0, 'email_count': 0, 'phone_count': 0,
            'has_url': 0, 'has_email': 0, 'has_phone': 0,
            'urgency_score': 0, 'threat_score': 0, 'credential_score': 0,
            'financial_score': 0, 'action_score': 0, 'impersonation_score': 0,
            'uppercase_ratio': 0, 'exclamation_count': 0, 'question_count': 0,
            'punctuation_ratio': 0, 'avg_word_length': 0,
            'short_word_ratio': 0, 'long_word_ratio': 0,
            'all_caps_word_count': 0, 'capitalized_word_ratio': 0,
            'has_currency': 0, 'has_percentage': 0, 'digit_ratio': 0,
            'phishing_indicator_score': 0
        }

    def _count_pattern_matches(self, text: str, patterns: List[re.Pattern]) -> int:
        """Count total matches for a list of patterns"""
        count = 0
        for pattern in patterns:
            count += len(pattern.findall(text))
        return count

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return list(self._empty_features().keys())


class CombinedFeatureExtractor:
    """
    Combined feature extractor for both URLs and text

    Extracts features from both URL and text content,
    useful when analyzing messages that contain URLs.
    """

    def __init__(self):
        """Initialize combined extractor"""
        self.url_extractor = URLFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()

    def extract(self, text: str, url: Optional[str] = None) -> Dict[str, float]:
        """
        Extract combined features from text and optional URL

        Args:
            text: Text content
            url: Optional specific URL to analyze

        Returns:
            Combined feature dictionary
        """
        features = {}

        # Extract text features
        text_features = self.text_extractor.extract(text)
        for key, value in text_features.items():
            features[f'text_{key}'] = value

        # Extract URL features from provided URL or first URL in text
        if url:
            url_features = self.url_extractor.extract(url)
        else:
            # Try to find URL in text
            url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
            urls = url_pattern.findall(text)
            if urls:
                url_features = self.url_extractor.extract(urls[0])
            else:
                url_features = self.url_extractor._empty_features()

        for key, value in url_features.items():
            features[f'url_{key}'] = value

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        text_names = [f'text_{n}' for n in self.text_extractor.get_feature_names()]
        url_names = [f'url_{n}' for n in self.url_extractor.get_feature_names()]
        return text_names + url_names


# Convenience functions
def extract_url_features(url: str) -> Dict[str, float]:
    """Extract features from a URL"""
    extractor = URLFeatureExtractor()
    return extractor.extract(url)


def extract_text_features(text: str) -> Dict[str, float]:
    """Extract features from text"""
    extractor = TextFeatureExtractor()
    return extractor.extract(text)


def extract_combined_features(text: str, url: Optional[str] = None) -> Dict[str, float]:
    """Extract combined features from text and URL"""
    extractor = CombinedFeatureExtractor()
    return extractor.extract(text, url)
