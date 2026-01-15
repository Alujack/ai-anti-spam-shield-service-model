"""
Phishing Detection Model Training Pipeline
Trains ML models for phishing, smishing, and URL-based attack detection

Supports multiple model types:
- XGBoost: Feature-based classification (fast, interpretable)
- Random Forest: Ensemble tree-based classification
- Logistic Regression: Baseline linear model
- BERT: Transformer-based text classification (optional, requires GPU)

Usage:
    python train_phishing.py --data datasets/prepared/train.csv --model xgboost
    python train_phishing.py --data datasets/prepared --all-splits --model random_forest

Requirements:
    pip install scikit-learn xgboost pandas numpy joblib
    pip install transformers torch (for BERT)
"""

import os
import sys
import csv
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    import joblib
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install scikit-learn pandas numpy joblib")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not installed, using Random Forest instead")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.model.feature_extractors import (
        URLFeatureExtractor,
        TextFeatureExtractor,
        CombinedFeatureExtractor
    )
except ImportError:
    # Fallback for direct execution
    from feature_extractors import (
        URLFeatureExtractor,
        TextFeatureExtractor,
        CombinedFeatureExtractor
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingModelTrainer:
    """Train ML models for phishing detection"""

    SUPPORTED_MODELS = ['xgboost', 'random_forest', 'logistic_regression']

    def __init__(self, model_type: str = 'xgboost', model_dir: str = 'models'):
        """
        Initialize trainer

        Args:
            model_type: Type of model to train
            model_dir: Directory to save trained models
        """
        self.model_type = model_type
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.url_extractor = URLFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.combined_extractor = CombinedFeatureExtractor()

        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.training_stats = {}

    def load_data(self, filepath: str) -> Tuple[List[str], List[int]]:
        """
        Load training data from CSV

        Args:
            filepath: Path to CSV file with 'text' and 'label' columns

        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Loading data from {filepath}")

        texts = []
        labels = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)

            for row in reader:
                text = row.get('text', '').strip()
                label = row.get('label', '').strip().lower()

                if not text:
                    continue

                # Convert label to binary
                if label in ['phishing', 'spam', 'smishing', 'scam', 'malicious']:
                    labels.append(1)
                elif label in ['legitimate', 'ham', 'safe', 'benign']:
                    labels.append(0)
                else:
                    continue

                texts.append(text)

        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Phishing: {sum(labels)}, Legitimate: {len(labels) - sum(labels)}")

        return texts, labels

    def extract_features(self, texts: List[str], use_tfidf: bool = True) -> np.ndarray:
        """
        Extract features from texts

        Args:
            texts: List of text samples
            use_tfidf: Whether to include TF-IDF features

        Returns:
            Feature matrix
        """
        logger.info("Extracting features...")

        # Extract custom features
        custom_features = []
        for text in texts:
            features = self.combined_extractor.extract(text)
            custom_features.append(list(features.values()))

        custom_array = np.array(custom_features)

        if use_tfidf:
            # Fit or transform with TF-IDF
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    max_features=3000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8,
                    strip_accents='unicode',
                    lowercase=True
                )
                tfidf_array = self.vectorizer.fit_transform(texts).toarray()
            else:
                tfidf_array = self.vectorizer.transform(texts).toarray()

            # Combine features
            features = np.hstack([custom_array, tfidf_array])
        else:
            features = custom_array

        # Store feature names
        if self.feature_names is None:
            custom_names = self.combined_extractor.get_feature_names()
            if use_tfidf:
                tfidf_names = [f'tfidf_{i}' for i in range(tfidf_array.shape[1])]
                self.feature_names = custom_names + tfidf_names
            else:
                self.feature_names = custom_names

        logger.info(f"Extracted {features.shape[1]} features")
        return features

    def create_model(self) -> Any:
        """
        Create the ML model based on model_type

        Returns:
            Untrained model instance
        """
        if self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                logger.warning("XGBoost not available, using Random Forest")
                self.model_type = 'random_forest'
            else:
                return XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )

        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        if self.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
                n_jobs=-1
            )

        raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: List[int],
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[List[int]] = None) -> Dict:
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training statistics
        """
        logger.info(f"Training {self.model_type} model...")

        self.model = self.create_model()

        # If no validation set, split training data
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
            )

        # Train
        start_time = datetime.now()

        if self.model_type == 'xgboost' and HAS_XGBOOST:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        train_time = (datetime.now() - start_time).total_seconds()

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]

        stats = {
            'model_type': self.model_type,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'train_time_seconds': train_time,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_prob),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }

        self.training_stats = stats

        logger.info(f"Training complete in {train_time:.1f}s")
        logger.info(f"Accuracy: {stats['accuracy']:.4f}")
        logger.info(f"Precision: {stats['precision']:.4f}")
        logger.info(f"Recall: {stats['recall']:.4f}")
        logger.info(f"F1 Score: {stats['f1']:.4f}")
        logger.info(f"ROC AUC: {stats['roc_auc']:.4f}")

        return stats

    def evaluate(self, X_test: np.ndarray, y_test: List[int]) -> Dict:
        """
        Evaluate model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating on test set...")

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'test_samples': len(y_test),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1 Score: {metrics['f1']:.4f}")
        logger.info(f"Test ROC AUC: {metrics['roc_auc']:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importance from trained model

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.model is None:
            return []

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return []

        if self.feature_names is None:
            return []

        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]

        return [(self.feature_names[i], importances[i]) for i in indices]

    def save_model(self, filename: str = 'phishing_model') -> Dict[str, str]:
        """
        Save trained model and vectorizer

        Args:
            filename: Base filename for model files

        Returns:
            Dictionary of saved file paths
        """
        paths = {}

        # Save model
        model_path = self.model_dir / f'{filename}.pkl'
        joblib.dump(self.model, model_path)
        paths['model'] = str(model_path)
        logger.info(f"Saved model to {model_path}")

        # Save vectorizer
        if self.vectorizer is not None:
            vectorizer_path = self.model_dir / f'{filename}_vectorizer.pkl'
            joblib.dump(self.vectorizer, vectorizer_path)
            paths['vectorizer'] = str(vectorizer_path)
            logger.info(f"Saved vectorizer to {vectorizer_path}")

        # Save feature names
        features_path = self.model_dir / f'{filename}_features.json'
        with open(features_path, 'w') as f:
            json.dump({
                'feature_names': self.feature_names,
                'custom_features': self.combined_extractor.get_feature_names()
            }, f, indent=2)
        paths['features'] = str(features_path)

        # Save training stats
        stats_path = self.model_dir / f'{filename}_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)
        paths['stats'] = str(stats_path)

        return paths

    def load_model(self, filename: str = 'phishing_model') -> bool:
        """
        Load trained model and vectorizer

        Args:
            filename: Base filename for model files

        Returns:
            True if loaded successfully
        """
        try:
            model_path = self.model_dir / f'{filename}.pkl'
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")

            vectorizer_path = self.model_dir / f'{filename}_vectorizer.pkl'
            if vectorizer_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info(f"Loaded vectorizer from {vectorizer_path}")

            features_path = self.model_dir / f'{filename}_features.json'
            if features_path.exists():
                with open(features_path, 'r') as f:
                    data = json.load(f)
                    self.feature_names = data.get('feature_names')

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def predict(self, text: str) -> Dict:
        """
        Predict if text is phishing

        Args:
            text: Text to analyze

        Returns:
            Prediction result
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        # Extract features
        features = self.extract_features([text], use_tfidf=self.vectorizer is not None)

        # Predict
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]

        return {
            'is_phishing': bool(prediction),
            'confidence': float(probability[1]),
            'legitimate_probability': float(probability[0]),
            'phishing_probability': float(probability[1])
        }


def main():
    parser = argparse.ArgumentParser(
        description='Train phishing detection models'
    )
    parser.add_argument(
        '--data',
        required=True,
        help='Path to training CSV or directory with train/val/test splits'
    )
    parser.add_argument(
        '--model',
        choices=['xgboost', 'random_forest', 'logistic_regression'],
        default='xgboost',
        help='Model type to train (default: xgboost)'
    )
    parser.add_argument(
        '--output',
        default='models',
        help='Output directory for trained models'
    )
    parser.add_argument(
        '--all-splits',
        action='store_true',
        help='Use separate train/val/test files from directory'
    )
    parser.add_argument(
        '--name',
        default='phishing_model',
        help='Name for saved model files'
    )

    args = parser.parse_args()

    trainer = PhishingModelTrainer(
        model_type=args.model,
        model_dir=args.output
    )

    if args.all_splits:
        # Load from directory with separate files
        data_dir = Path(args.data)

        train_file = data_dir / 'train.csv'
        val_file = data_dir / 'validation.csv'
        test_file = data_dir / 'test.csv'

        if not train_file.exists():
            print(f"Train file not found: {train_file}")
            sys.exit(1)

        # Load training data
        train_texts, train_labels = trainer.load_data(str(train_file))
        X_train = trainer.extract_features(train_texts)
        y_train = train_labels

        # Load validation data
        X_val, y_val = None, None
        if val_file.exists():
            val_texts, val_labels = trainer.load_data(str(val_file))
            X_val = trainer.extract_features(val_texts)
            y_val = val_labels

        # Train
        trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        if test_file.exists():
            test_texts, test_labels = trainer.load_data(str(test_file))
            X_test = trainer.extract_features(test_texts)
            trainer.evaluate(X_test, test_labels)

    else:
        # Load single file and split
        texts, labels = trainer.load_data(args.data)
        X = trainer.extract_features(texts)
        y = labels

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train
        trainer.train(X_train, y_train)

        # Evaluate
        trainer.evaluate(X_test, y_test)

    # Print feature importance
    print("\nTop 20 Feature Importances:")
    for name, importance in trainer.get_feature_importance(20):
        print(f"  {name}: {importance:.4f}")

    # Save model
    paths = trainer.save_model(args.name)
    print(f"\nModel saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
