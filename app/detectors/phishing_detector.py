"""
Enhanced Phishing Detector with ML-based Detection
Supports: Email phishing, SMS phishing (smishing), URL phishing

This module provides production-grade phishing detection using:
- ML-based ensemble classification (XGBoost, Random Forest)
- Rule-based pattern matching for explainability
- URL analysis with suspicious pattern detection
- Brand impersonation detection
- Pre-trained transformer fallback (optional)

Trained on: PhishTank, OpenPhish, Enron, UCI SMS, Tranco (legitimate)
"""

import re
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

try:
    import tldextract
    HAS_TLDEXTRACT = True
except ImportError:
    HAS_TLDEXTRACT = False

try:
    import joblib
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Import feature extractors
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from model.feature_extractors import (
        URLFeatureExtractor,
        TextFeatureExtractor,
        CombinedFeatureExtractor
    )
    HAS_EXTRACTORS = True
except ImportError:
    HAS_EXTRACTORS = False

logger = logging.getLogger(__name__)


class PhishingType(Enum):
    """Types of phishing attacks"""
    EMAIL = "EMAIL"
    SMS = "SMS"
    URL = "URL"
    NONE = "NONE"


class ThreatLevel(Enum):
    """Threat severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass
class URLAnalysisResult:
    """Result of analyzing a single URL"""
    url: str
    is_suspicious: bool
    score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BrandImpersonation:
    """Brand impersonation detection result"""
    detected: bool
    brand: Optional[str] = None
    similarity_score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PhishingResult:
    """Complete phishing detection result"""
    is_phishing: bool
    confidence: float
    phishing_type: str
    threat_level: str
    indicators: List[str]
    urls_analyzed: List[Dict]
    brand_impersonation: Optional[Dict]
    recommendation: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class PhishingDetector:
    """
    Enhanced phishing detection with ML ensemble

    Combines rule-based detection with ML models for
    accurate and explainable phishing detection.
    """

    # Known brands commonly impersonated
    KNOWN_BRANDS = [
        'paypal', 'amazon', 'netflix', 'apple', 'microsoft', 'google',
        'facebook', 'instagram', 'whatsapp', 'chase', 'wellsfargo',
        'bankofamerica', 'citibank', 'usps', 'fedex', 'ups', 'dhl',
        'irs', 'ssa', 'walmart', 'target', 'ebay', 'linkedin',
        'dropbox', 'adobe', 'spotify', 'yahoo', 'outlook', 'office365'
    ]

    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = [
        'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work', 'click',
        'link', 'info', 'online', 'site', 'website', 'space', 'pw'
    ]

    # URL shorteners commonly abused
    URL_SHORTENERS = [
        'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'j.mp', 'tr.im', 'cutt.ly', 'rb.gy'
    ]

    # Urgency patterns
    URGENCY_PATTERNS = [
        r'\b(urgent|immediately|asap|now|today|expires?)\b',
        r'\b(act now|hurry|don\'t wait|final notice|last chance)\b',
        r'\b(within \d+ (hours?|days?|minutes?))\b',
        r'\b(account (suspended|locked|compromised|restricted))\b',
        r'\b(verify (your )?(account|identity|password))\b',
    ]

    # Credential request patterns
    CREDENTIAL_PATTERNS = [
        r'\b(enter (your )?(password|pin|ssn|credit card))\b',
        r'\b(confirm (your )?(account|identity|details))\b',
        r'\b(update (your )?(information|password|credentials))\b',
    ]

    # Financial patterns
    FINANCIAL_PATTERNS = [
        r'\b(bank|credit card|debit card|payment|transaction)\b',
        r'\b(wire transfer|bitcoin|cryptocurrency)\b',
        r'[$£€¥]\s?\d+[,\d]*',
    ]

    def __init__(self, model_dir: str = 'models', use_ml: bool = True):
        """
        Initialize the enhanced phishing detector

        Args:
            model_dir: Directory containing trained ML models
            use_ml: Whether to use ML models (if available)
        """
        self.model_dir = Path(model_dir)
        self.use_ml = use_ml and HAS_ML

        # Initialize ML components
        self.ml_model = None
        self.vectorizer = None
        self.transformer_model = None

        # Initialize feature extractors
        if HAS_EXTRACTORS:
            self.url_extractor = URLFeatureExtractor()
            self.text_extractor = TextFeatureExtractor()
            self.combined_extractor = CombinedFeatureExtractor()
        else:
            self.url_extractor = None
            self.text_extractor = None
            self.combined_extractor = None

        # Compile patterns
        self.urgency_patterns = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS]
        self.credential_patterns = [re.compile(p, re.IGNORECASE) for p in self.CREDENTIAL_PATTERNS]
        self.financial_patterns = [re.compile(p, re.IGNORECASE) for p in self.FINANCIAL_PATTERNS]

        # Brand pattern
        self.brand_pattern = re.compile(
            r'\b(' + '|'.join(self.KNOWN_BRANDS) + r')\b',
            re.IGNORECASE
        )

        # URL pattern
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE
        )

        # Load ML models
        if self.use_ml:
            self._load_ml_models()

        # Load transformer model (optional)
        if HAS_TRANSFORMERS:
            self._load_transformer_model()

        logger.info(f"PhishingDetector initialized (ML: {self.ml_model is not None})")

    def _load_ml_models(self):
        """Load trained ML models"""
        try:
            model_path = self.model_dir / 'phishing_model.pkl'
            vectorizer_path = self.model_dir / 'phishing_model_vectorizer.pkl'

            if model_path.exists():
                self.ml_model = joblib.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")

            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info(f"Loaded vectorizer from {vectorizer_path}")

        except Exception as e:
            logger.warning(f"Failed to load ML models: {e}")
            self.ml_model = None
            self.vectorizer = None

    def _load_transformer_model(self):
        """Load pre-trained transformer model as fallback"""
        try:
            self.transformer_model = pipeline(
                "text-classification",
                model="ealvaradob/bert-finetuned-phishing",
                truncation=True,
                max_length=512
            )
            logger.info("Loaded transformer model for phishing detection")
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            self.transformer_model = None

    def detect(self, text: str, scan_type: str = 'auto') -> PhishingResult:
        """
        Main phishing detection method

        Args:
            text: Message text or URL to analyze
            scan_type: 'email', 'sms', 'url', or 'auto'

        Returns:
            PhishingResult with comprehensive analysis
        """
        if not text or len(text.strip()) < 3:
            return self._empty_result()

        text = text.strip()

        # Auto-detect scan type
        if scan_type == 'auto':
            scan_type = self._detect_scan_type(text)

        # Extract URLs
        urls = self.extract_urls(text)

        # Run all detection methods
        ml_score = self._ml_detection(text)
        rule_score, rule_indicators = self._rule_based_detection(text)
        url_results = self._analyze_urls(urls)
        brand_result = self._detect_brand_impersonation(text, urls)
        transformer_score = self._transformer_detection(text)

        # Calculate ensemble score
        final_score = self._ensemble_score(
            ml_score,
            rule_score,
            url_results,
            brand_result,
            transformer_score
        )

        # Collect all indicators
        indicators = self._collect_indicators(
            text, urls, brand_result, rule_indicators
        )

        # Determine threat level
        threat_level = self._determine_threat_level(final_score, indicators)

        # Determine phishing type
        phishing_type = self._determine_phishing_type(
            scan_type, indicators, url_results
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            final_score, phishing_type, threat_level
        )

        return PhishingResult(
            is_phishing=final_score >= 0.5,
            confidence=round(final_score, 4),
            phishing_type=phishing_type.value,
            threat_level=threat_level.value,
            indicators=indicators,
            urls_analyzed=[u.to_dict() for u in url_results],
            brand_impersonation=brand_result.to_dict() if brand_result.detected else None,
            recommendation=recommendation,
            details={
                'ml_score': ml_score,
                'rule_score': rule_score,
                'transformer_score': transformer_score,
                'url_count': len(urls),
                'scan_type': scan_type
            }
        )

    def _empty_result(self) -> PhishingResult:
        """Return empty/safe result"""
        return PhishingResult(
            is_phishing=False,
            confidence=0.0,
            phishing_type=PhishingType.NONE.value,
            threat_level=ThreatLevel.NONE.value,
            indicators=[],
            urls_analyzed=[],
            brand_impersonation=None,
            recommendation="No suspicious content detected.",
            details={}
        )

    def _detect_scan_type(self, text: str) -> str:
        """Auto-detect the type of content"""
        text_lower = text.lower()

        # Check if it's primarily a URL
        if text.startswith('http://') or text.startswith('https://'):
            if len(text.split()) <= 3:
                return 'url'

        # Check for SMS patterns
        sms_patterns = [
            r'\b(txt|sms|text me)\b',
            r'\b(click link|tap here)\b',
            len(text) < 500  # SMS are typically short
        ]
        sms_score = sum(1 for p in sms_patterns[:2] if re.search(p, text_lower, re.IGNORECASE))
        if sms_score >= 1 and len(text) < 500:
            return 'sms'

        # Default to email for longer content
        return 'email'

    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        return self.url_pattern.findall(text)

    def _ml_detection(self, text: str) -> float:
        """Use trained ML model for detection"""
        if self.ml_model is None or self.combined_extractor is None:
            return 0.5  # Neutral if no model

        try:
            # Extract features
            features = self.combined_extractor.extract(text)
            feature_values = list(features.values())

            # Add TF-IDF features if available
            if self.vectorizer is not None:
                tfidf_features = self.vectorizer.transform([text]).toarray()[0]
                feature_values.extend(tfidf_features)

            # Predict
            X = np.array([feature_values])
            probability = self.ml_model.predict_proba(X)[0]

            return float(probability[1])  # Probability of phishing

        except Exception as e:
            logger.warning(f"ML detection failed: {e}")
            return 0.5

    def _transformer_detection(self, text: str) -> float:
        """Use transformer model for detection"""
        if self.transformer_model is None:
            return 0.5  # Neutral if no model

        try:
            result = self.transformer_model(text[:512])[0]  # Truncate for BERT

            # Map label to score
            if result['label'].lower() in ['phishing', 'spam', '1', 'positive']:
                return result['score']
            else:
                return 1.0 - result['score']

        except Exception as e:
            logger.warning(f"Transformer detection failed: {e}")
            return 0.5

    def _rule_based_detection(self, text: str) -> Tuple[float, List[str]]:
        """Rule-based detection for explainability"""
        indicators = []
        score = 0.0

        # Check urgency patterns
        urgency_count = sum(
            len(p.findall(text)) for p in self.urgency_patterns
        )
        if urgency_count > 0:
            score += min(0.25, urgency_count * 0.1)
            indicators.append(f"Urgency language detected ({urgency_count} patterns)")

        # Check credential patterns
        credential_count = sum(
            len(p.findall(text)) for p in self.credential_patterns
        )
        if credential_count > 0:
            score += min(0.3, credential_count * 0.15)
            indicators.append(f"Credential request detected ({credential_count} patterns)")

        # Check financial patterns
        financial_count = sum(
            len(p.findall(text)) for p in self.financial_patterns
        )
        if financial_count > 0:
            score += min(0.2, financial_count * 0.1)
            indicators.append(f"Financial keywords detected ({financial_count} patterns)")

        # Check for suspicious phrases
        suspicious_phrases = [
            'click here', 'verify your', 'confirm your', 'update your',
            'unusual activity', 'security alert', 'account locked',
            'winner', 'congratulations', 'prize', 'free gift'
        ]
        phrase_count = sum(1 for p in suspicious_phrases if p in text.lower())
        if phrase_count > 0:
            score += min(0.25, phrase_count * 0.08)
            indicators.append(f"Suspicious phrases detected ({phrase_count})")

        return min(score, 1.0), indicators

    def _analyze_urls(self, urls: List[str]) -> List[URLAnalysisResult]:
        """Analyze extracted URLs"""
        results = []

        for url in urls:
            reasons = []
            score = 0.0

            # Check for IP address
            if re.match(r'https?://\d+\.\d+\.\d+\.\d+', url):
                score += 0.4
                reasons.append("URL uses IP address instead of domain")

            # Check TLD
            if HAS_TLDEXTRACT:
                extracted = tldextract.extract(url)
                if extracted.suffix.lower() in self.SUSPICIOUS_TLDS:
                    score += 0.3
                    reasons.append(f"Suspicious TLD: .{extracted.suffix}")

                # Check for brand in subdomain (typosquatting)
                if self.brand_pattern.search(extracted.subdomain):
                    score += 0.3
                    reasons.append("Brand name in subdomain (possible typosquatting)")

            # Check for URL shortener
            url_lower = url.lower()
            if any(shortener in url_lower for shortener in self.URL_SHORTENERS):
                score += 0.2
                reasons.append("URL shortener detected")

            # Check for @ symbol (URL obfuscation)
            if '@' in url:
                score += 0.4
                reasons.append("URL contains @ symbol (possible obfuscation)")

            # Check for excessive subdomains
            if url.count('.') > 4:
                score += 0.2
                reasons.append("Excessive subdomains")

            # Check for suspicious keywords in URL
            suspicious_url_keywords = ['login', 'signin', 'verify', 'secure', 'account', 'update']
            keyword_count = sum(1 for kw in suspicious_url_keywords if kw in url_lower)
            if keyword_count >= 2:
                score += 0.2
                reasons.append(f"Suspicious keywords in URL ({keyword_count})")

            results.append(URLAnalysisResult(
                url=url,
                is_suspicious=score >= 0.3,
                score=min(score, 1.0),
                reasons=reasons
            ))

        return results

    def _detect_brand_impersonation(self, text: str, urls: List[str]) -> BrandImpersonation:
        """Detect brand impersonation attempts"""
        text_combined = text + ' ' + ' '.join(urls)
        matches = self.brand_pattern.findall(text_combined.lower())

        if not matches:
            return BrandImpersonation(detected=False)

        # Get most common brand mentioned
        brand = max(set(matches), key=matches.count)

        # Check if it's likely impersonation (brand mentioned but not official domain)
        is_impersonation = False
        similarity = 0.5

        for url in urls:
            if HAS_TLDEXTRACT:
                extracted = tldextract.extract(url)
                domain = f"{extracted.domain}.{extracted.suffix}".lower()

                # If brand is in text but domain doesn't match official brand
                if brand in text.lower() and brand not in domain:
                    is_impersonation = True
                    similarity = 0.8

        return BrandImpersonation(
            detected=is_impersonation,
            brand=brand if is_impersonation else None,
            similarity_score=similarity
        )

    def _ensemble_score(self, ml_score: float, rule_score: float,
                        url_results: List[URLAnalysisResult],
                        brand_result: BrandImpersonation,
                        transformer_score: float) -> float:
        """Calculate weighted ensemble score"""

        # URL score
        url_score = 0.0
        if url_results:
            url_score = max(r.score for r in url_results)

        # Brand score
        brand_score = brand_result.similarity_score if brand_result.detected else 0.0

        # Weights for ensemble
        weights = {
            'ml': 0.30,
            'rule': 0.25,
            'url': 0.20,
            'brand': 0.10,
            'transformer': 0.15
        }

        # Calculate weighted average
        score = (
            weights['ml'] * ml_score +
            weights['rule'] * rule_score +
            weights['url'] * url_score +
            weights['brand'] * brand_score +
            weights['transformer'] * transformer_score
        )

        # Boost score if multiple indicators present
        indicator_count = sum([
            ml_score > 0.6,
            rule_score > 0.3,
            url_score > 0.3,
            brand_result.detected,
            transformer_score > 0.6
        ])

        if indicator_count >= 3:
            score = min(score * 1.2, 1.0)

        return min(score, 1.0)

    def _collect_indicators(self, text: str, urls: List[str],
                            brand_result: BrandImpersonation,
                            rule_indicators: List[str]) -> List[str]:
        """Collect all detected indicators"""
        indicators = list(rule_indicators)

        # Add URL indicators
        for url in urls:
            if HAS_TLDEXTRACT:
                extracted = tldextract.extract(url)
                if extracted.suffix.lower() in self.SUSPICIOUS_TLDS:
                    indicators.append(f"Suspicious domain: {extracted.domain}.{extracted.suffix}")

        # Add brand impersonation
        if brand_result.detected:
            indicators.append(f"Possible {brand_result.brand} impersonation detected")

        # Add URL shortener warning
        for url in urls:
            if any(s in url.lower() for s in self.URL_SHORTENERS):
                indicators.append("URL shortener used (may hide destination)")
                break

        return list(set(indicators))  # Remove duplicates

    def _determine_threat_level(self, score: float, indicators: List[str]) -> ThreatLevel:
        """Determine threat level based on score and indicators"""
        if score >= 0.85 or len(indicators) >= 5:
            return ThreatLevel.CRITICAL
        elif score >= 0.7 or len(indicators) >= 3:
            return ThreatLevel.HIGH
        elif score >= 0.5 or len(indicators) >= 2:
            return ThreatLevel.MEDIUM
        elif score >= 0.3 or len(indicators) >= 1:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE

    def _determine_phishing_type(self, scan_type: str, indicators: List[str],
                                  url_results: List[URLAnalysisResult]) -> PhishingType:
        """Determine the type of phishing attack"""
        if scan_type == 'url' or (url_results and all(r.is_suspicious for r in url_results)):
            return PhishingType.URL
        elif scan_type == 'sms':
            return PhishingType.SMS
        elif scan_type == 'email':
            return PhishingType.EMAIL
        elif url_results and any(r.is_suspicious for r in url_results):
            return PhishingType.URL
        else:
            return PhishingType.NONE

    def _generate_recommendation(self, score: float, phishing_type: PhishingType,
                                  threat_level: ThreatLevel) -> str:
        """Generate actionable recommendation"""
        if threat_level == ThreatLevel.NONE:
            return "This message appears to be safe. However, always verify sender identity for sensitive requests."

        if threat_level == ThreatLevel.LOW:
            return "Some suspicious elements detected. Verify the sender before taking any action."

        if threat_level == ThreatLevel.MEDIUM:
            return "This message contains suspicious content. Do not click links or provide personal information. Verify directly with the organization."

        if threat_level == ThreatLevel.HIGH:
            return "This message is likely a phishing attempt. Do not respond, click links, or provide any information. Report and delete."

        if threat_level == ThreatLevel.CRITICAL:
            return "DANGER: This is a phishing attack. Do not interact with this message in any way. Report to your IT department and delete immediately."

        return "Exercise caution with this message."

    # Legacy methods for backward compatibility
    def is_suspicious_domain(self, url: str) -> bool:
        """Check if domain is suspicious (legacy method)"""
        if HAS_TLDEXTRACT:
            extracted = tldextract.extract(url)
            return extracted.suffix in self.SUSPICIOUS_TLDS
        return False

    def check_suspicious_urls(self, urls: List[str]) -> bool:
        """Check for suspicious URLs (legacy method)"""
        return any(self.is_suspicious_domain(url) for url in urls)

    def check_brand_impersonation(self, text: str) -> bool:
        """Check for brand impersonation (legacy method)"""
        return bool(self.brand_pattern.search(text))

    def check_urgency_language(self, text: str) -> bool:
        """Check for urgency language (legacy method)"""
        return any(p.search(text) for p in self.urgency_patterns)
