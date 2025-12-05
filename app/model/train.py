import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        self.model = None
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
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
    
    def load_data(self, filepath):
        """Load dataset from file"""
        data = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    label, message = line.strip().split('\t', 1)
                    data.append(message)
                    labels.append(1 if label == 'spam' else 0)
        
        return pd.DataFrame({
            'message': data,
            'label': labels
        })
    
    def train(self, X_train, y_train, model_type='naive_bayes'):
        """Train the model"""
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def predict(self, text):
        """Predict if text is spam or not"""
        processed_text = self.preprocess_text(text)
        text_vectorized = self.vectorizer.transform([processed_text])
        
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return {
            'is_spam': bool(prediction),
            'prediction': 'spam' if prediction else 'ham',
            'confidence': float(probability[1] if prediction else probability[0]),
            'probabilities': {
                'ham': float(probability[0]),
                'spam': float(probability[1])
            }
        }
    
    def save_model(self, model_dir='model'):
        """Save trained model and vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'spam_classifier.pkl'))
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.pkl'))
        
        print(f"Model saved to {model_dir}/")
    
    def load_model(self, model_dir='model'):
        """Load trained model and vectorizer"""
        self.model = joblib.load(os.path.join(model_dir, 'spam_classifier.pkl'))
        self.vectorizer = joblib.load(os.path.join(model_dir, 'vectorizer.pkl'))
        
        print("Model loaded successfully!")

def train_spam_classifier(data_path, model_type='naive_bayes'):
    """
    Main training function
    """
    print("🚀 Starting Spam Classifier Training...")
    print("=" * 60)
    
    # Initialize classifier
    classifier = SpamClassifier()
    
    # Load data
    print("\n📊 Loading dataset...")
    df = classifier.load_data(data_path)
    print(f"Total samples: {len(df)}")
    print(f"Spam: {df['label'].sum()}, Ham: {len(df) - df['label'].sum()}")
    
    # Preprocess data
    print("\n🔄 Preprocessing text...")
    df['processed_message'] = df['message'].apply(classifier.preprocess_text)
    
    # Split data
    print("\n✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_message'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Vectorize text
    print("\n📐 Vectorizing text...")
    X_train_vectorized = classifier.vectorizer.fit_transform(X_train)
    X_test_vectorized = classifier.vectorizer.transform(X_test)
    print(f"Features: {X_train_vectorized.shape[1]}")
    
    # Train model
    print(f"\n🎯 Training {model_type} model...")
    classifier.train(X_train_vectorized, y_train, model_type=model_type)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    results = classifier.evaluate(X_test_vectorized, y_test)
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save model
    print("\n💾 Saving model...")
    classifier.save_model('model')
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been suspended. Verify now!",
        "Can you send me the project report by end of day?"
    ]
    
    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"\nMessage: {msg[:50]}...")
        print(f"Prediction: {result['prediction'].upper()} (confidence: {result['confidence']:.2%})")
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    
    return classifier

if __name__ == '__main__':
    import sys
    
    # Get data path
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '../datasets/spam_sample.txt'
    
    # Get model type
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'naive_bayes'
    
    # Train model
    classifier = train_spam_classifier(data_path, model_type)

