import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SpamPredictor:
    # Minimum confidence threshold to classify as spam
    # Messages below this threshold are considered "ham" even if model predicts spam
    # This reduces false positives for ambiguous messages
    SPAM_CONFIDENCE_THRESHOLD = 0.75  # 75%

    def __init__(self, model_dir='model'):
        """Initialize the predictor with trained model"""
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.stemmer = PorterStemmer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and vectorizer"""
        model_path = os.path.join(self.model_dir, 'spam_classifier.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Model files not found in {self.model_dir}. "
                "Please train the model first using train.py"
            )
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        
        print("✅ Model loaded successfully!")
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and apply stemming
        if self.stop_words:
            words = text.split()
            words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def extract_features(self, text):
        """Extract features from text for explainability"""
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_url': bool(re.search(r'http\S+|www\S+', text, re.IGNORECASE)),
            'has_email': bool(re.search(r'\S+@\S+', text)),
            'has_phone': bool(re.search(r'\b\d{10,}\b', text)),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'currency_symbols': bool(re.search(r'[$£€¥]', text)),
            'urgency_words': bool(re.search(r'\b(urgent|asap|now|immediately|hurry)\b', text, re.IGNORECASE)),
            'spam_keywords': bool(re.search(r'\b(free|win|winner|prize|claim|click|buy|offer|deal)\b', text, re.IGNORECASE)),
        }
        return features
    
    def predict(self, text):
        """Predict if text is spam or not"""
        # Preprocess
        processed_text = self.preprocess_text(text)

        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])

        # Predict
        raw_prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]

        # Extract features for explainability
        features = self.extract_features(text)

        # Apply confidence threshold to reduce false positives
        # Only classify as spam if confidence exceeds threshold
        spam_probability = float(probabilities[1])
        is_spam = spam_probability >= self.SPAM_CONFIDENCE_THRESHOLD

        # Prepare result
        result = {
            'is_spam': is_spam,
            'prediction': 'spam' if is_spam else 'ham',
            'confidence': spam_probability if is_spam else float(probabilities[0]),
            'probability': spam_probability,  # Spam probability
            'probabilities': {
                'ham': float(probabilities[0]),
                'spam': spam_probability
            },
            'details': {
                'features': features,
                'processed_text_length': len(processed_text),
                'original_text_length': len(text),
                'threshold_applied': self.SPAM_CONFIDENCE_THRESHOLD,
                'raw_prediction': 'spam' if raw_prediction else 'ham'
            }
        }

        return result
    
    def batch_predict(self, texts):
        """Predict multiple texts at once"""
        results = []
        for text in texts:
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'is_spam': False,
                    'prediction': 'error'
                })
        return results

