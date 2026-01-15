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


def load_huggingface_dataset(dataset_name="Deysi/spam-detection-dataset", split="train"):
    """
    Load spam detection dataset from Hugging Face

    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Which split to load ('train', 'test', or 'all')

    Returns:
        pandas DataFrame with 'message' and 'label' columns
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: pip install datasets"
        )

    print(f"📥 Loading dataset from Hugging Face: {dataset_name}")

    if split == "all":
        # Load both train and test splits
        dataset = load_dataset(dataset_name)
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()

    # Rename columns to match expected format
    # The Deysi/spam-detection-dataset has 'text' and 'label' columns
    # where label is 'spam' or 'not_spam'
    df = df.rename(columns={'text': 'message'})

    # Convert labels to binary (1 for spam, 0 for not_spam/ham)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

    # Drop the 'split' column if it exists
    if 'split' in df.columns:
        df = df.drop(columns=['split'])

    print(f"✅ Loaded {len(df)} samples from Hugging Face dataset")

    return df


def load_scam_dialogue_dataset(dataset_name="BothBosu/scam-dialogue", split="all"):
    """
    Load scam dialogue dataset from Hugging Face for voice call scam detection

    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Which split to load ('train', 'test', or 'all')

    Returns:
        pandas DataFrame with 'message' and 'label' columns
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' library: pip install datasets"
        )

    print(f"📥 Loading scam dialogue dataset from Hugging Face: {dataset_name}")

    if split == "all":
        # Load both train and test splits
        dataset = load_dataset(dataset_name)
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()
        df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        dataset = load_dataset(dataset_name, split=split)
        df = dataset.to_pandas()

    # Rename columns to match expected format
    # The BothBosu/scam-dialogue has 'dialogue', 'type', and 'label' columns
    # label is already binary (1 = scam, 0 = non-scam)
    df = df.rename(columns={'dialogue': 'message'})

    # Keep only message and label columns
    df = df[['message', 'label']]

    print(f"✅ Loaded {len(df)} scam dialogue samples")

    return df


def load_combined_datasets():
    """
    Load and combine both SMS spam and scam dialogue datasets for unified training

    Returns:
        pandas DataFrame with 'message' and 'label' columns containing both datasets
    """
    print("📥 Loading combined datasets for unified training...")
    print("=" * 60)

    # Load SMS spam dataset
    print("\n[1/2] Loading SMS spam dataset...")
    sms_df = load_huggingface_dataset("Deysi/spam-detection-dataset", split="all")
    sms_df['source'] = 'sms_spam'

    # Load scam dialogue dataset
    print("\n[2/2] Loading scam dialogue dataset...")
    dialogue_df = load_scam_dialogue_dataset("BothBosu/scam-dialogue", split="all")
    dialogue_df['source'] = 'scam_dialogue'

    # Combine datasets
    print("\n🔗 Combining datasets...")
    combined_df = pd.concat([sms_df, dialogue_df], ignore_index=True)

    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n📊 Combined Dataset Statistics:")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   SMS spam samples: {len(sms_df)}")
    print(f"   Scam dialogue samples: {len(dialogue_df)}")
    print(f"   Total spam/scam: {combined_df['label'].sum()}")
    print(f"   Total ham/non-scam: {len(combined_df) - combined_df['label'].sum()}")

    return combined_df


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

def train_from_huggingface(
    dataset_name="Deysi/spam-detection-dataset",
    model_type='naive_bayes',
    use_all_data=True
):
    """
    Train spam classifier using Hugging Face dataset

    Args:
        dataset_name: Name of the Hugging Face dataset
        model_type: 'naive_bayes', 'logistic_regression', or 'random_forest'
        use_all_data: If True, combines train and test splits for more training data

    Returns:
        Trained SpamClassifier instance
    """
    print("🚀 Starting Spam Classifier Training with Hugging Face Dataset...")
    print("=" * 60)

    # Initialize classifier
    classifier = SpamClassifier()

    # Load data from Hugging Face
    print("\n📊 Loading dataset from Hugging Face...")
    split = "all" if use_all_data else "train"
    df = load_huggingface_dataset(dataset_name, split=split)
    print(f"Total samples: {len(df)}")
    print(f"Spam: {df['label'].sum()}, Ham/Not Spam: {len(df) - df['label'].sum()}")

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
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    # Save model
    print("\n💾 Saving model...")
    classifier.save_model('model')

    # Test predictions
    print("\n🧪 Testing predictions...")
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been suspended. Verify now!",
        "Can you send me the project report by end of day?",
        "Make $5000 per week working from home! Limited spots available!",
        "Thanks for your help yesterday, really appreciate it."
    ]

    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"\nMessage: {msg[:60]}...")
        print(f"Prediction: {result['prediction'].upper()} (confidence: {result['confidence']:.2%})")

    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")

    return classifier


def train_scam_dialogue_classifier(model_type='naive_bayes'):
    """
    Train classifier using only the scam dialogue dataset

    Args:
        model_type: 'naive_bayes', 'logistic_regression', or 'random_forest'

    Returns:
        Trained SpamClassifier instance
    """
    print("🚀 Starting Scam Dialogue Classifier Training...")
    print("=" * 60)

    # Initialize classifier with higher max_features for longer dialogues
    classifier = SpamClassifier()
    classifier.vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 3)
    )

    # Load scam dialogue data
    print("\n📊 Loading scam dialogue dataset...")
    df = load_scam_dialogue_dataset("BothBosu/scam-dialogue", split="all")
    print(f"Total samples: {len(df)}")
    print(f"Scam: {df['label'].sum()}, Non-scam: {len(df) - df['label'].sum()}")

    # Preprocess data
    print("\n🔄 Preprocessing dialogues...")
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
    print("\n📐 Vectorizing dialogues...")
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
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    # Save model
    print("\n💾 Saving model...")
    classifier.save_model('model')

    # Test predictions with phone scam examples
    print("\n🧪 Testing predictions with phone scam examples...")
    test_messages = [
        "Hello, this is the IRS calling. You owe back taxes and must pay immediately or face arrest.",
        "Hi, this is Amazon customer service. Your package will be delivered tomorrow between 2-4 PM.",
        "Congratulations! You've won a free cruise. Press 1 to claim your prize now!",
        "This is your bank calling about suspicious activity. Please verify your account by providing your SSN.",
        "Hey, it's me. Just calling to check if you're still coming to dinner tonight.",
        "Your computer has been infected with a virus. Press 1 to speak with Microsoft support immediately."
    ]

    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"\nDialogue: {msg[:70]}...")
        print(f"Prediction: {result['prediction'].upper()} (confidence: {result['confidence']:.2%})")

    print("\n" + "=" * 60)
    print("✅ Scam dialogue training completed successfully!")

    return classifier


def train_unified_classifier(model_type='naive_bayes'):
    """
    Train unified classifier using both SMS spam and scam dialogue datasets

    This creates a single model that can detect:
    - SMS/text message spam
    - Voice call scams (from transcribed audio)

    Args:
        model_type: 'naive_bayes', 'logistic_regression', or 'random_forest'

    Returns:
        Trained SpamClassifier instance
    """
    print("🚀 Starting Unified Spam/Scam Classifier Training...")
    print("=" * 60)

    # Initialize classifier with higher max_features for combined dataset
    classifier = SpamClassifier()
    classifier.vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.85,
        ngram_range=(1, 3)
    )

    # Load combined datasets
    df = load_combined_datasets()

    # Preprocess data
    print("\n🔄 Preprocessing all text data...")
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
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    # Save model
    print("\n💾 Saving model...")
    classifier.save_model('model')

    # Test predictions with mixed examples
    print("\n🧪 Testing predictions with mixed examples...")
    test_messages = [
        # SMS spam examples
        "Congratulations! You've won a free iPhone. Click here to claim.",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account has been suspended. Verify now!",
        # Voice call scam examples
        "Hello, this is the IRS calling. You owe back taxes and must pay immediately.",
        "Hi, this is your doctor's office confirming your appointment for tomorrow.",
        "Your computer has a virus. Press 1 to speak with Microsoft support.",
        # Normal messages
        "Can you pick up some milk on your way home?",
        "Thanks for your help yesterday, really appreciate it."
    ]

    for msg in test_messages:
        result = classifier.predict(msg)
        print(f"\nMessage: {msg[:65]}...")
        print(f"Prediction: {result['prediction'].upper()} (confidence: {result['confidence']:.2%})")

    print("\n" + "=" * 60)
    print("✅ Unified training completed successfully!")
    print("📱 Model can now detect both SMS spam and voice call scams!")

    return classifier


if __name__ == '__main__':
    import sys

    # Print usage if --help
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
Usage: python train.py [OPTIONS] [MODEL_TYPE]

Training Options:
  --unified, -u        Train unified model with BOTH SMS spam and scam dialogues (RECOMMENDED)
  --scam-dialogue, -sd Train with scam dialogue dataset only (voice call scams)
  --huggingface, -hf   Train with SMS spam dataset only
  (default)            Train with local dataset file

Model Types:
  naive_bayes          Naive Bayes classifier (default, fastest)
  logistic_regression  Logistic Regression (balanced)
  random_forest        Random Forest (most accurate)

Examples:
  python train.py --unified                    # Train unified model (recommended)
  python train.py --unified logistic_regression
  python train.py --scam-dialogue              # Train scam dialogue only
  python train.py --huggingface                # Train SMS spam only
  python train.py ../datasets/spam_sample.txt  # Train with local file
        """)
        sys.exit(0)

    # Check for training mode flags
    use_unified = '--unified' in sys.argv or '-u' in sys.argv
    use_scam_dialogue = '--scam-dialogue' in sys.argv or '-sd' in sys.argv
    use_huggingface = '--huggingface' in sys.argv or '-hf' in sys.argv

    # Get model type
    model_type = 'naive_bayes'
    for arg in sys.argv[1:]:
        if arg in ['naive_bayes', 'logistic_regression', 'random_forest']:
            model_type = arg
            break

    if use_unified:
        # Train unified model with both datasets
        print("🔗 Training UNIFIED model with SMS spam + Scam dialogue datasets")
        classifier = train_unified_classifier(model_type=model_type)
    elif use_scam_dialogue:
        # Train with scam dialogue dataset only
        print("📞 Training with scam dialogue dataset: BothBosu/scam-dialogue")
        classifier = train_scam_dialogue_classifier(model_type=model_type)
    elif use_huggingface:
        # Train using SMS spam dataset
        print("📱 Training with SMS spam dataset: Deysi/spam-detection-dataset")
        classifier = train_from_huggingface(
            dataset_name="Deysi/spam-detection-dataset",
            model_type=model_type,
            use_all_data=True
        )
    else:
        # Use local file
        data_path = '../datasets/spam_sample.txt'
        for arg in sys.argv[1:]:
            if arg.endswith('.txt') or '/' in arg:
                data_path = arg
                break

        print(f"📄 Training with local dataset: {data_path}")
        classifier = train_spam_classifier(data_path, model_type)

