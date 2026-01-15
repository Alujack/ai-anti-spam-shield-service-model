"""
Dataset Balancer and Splitter
Balances phishing vs legitimate samples and creates train/val/test splits

This script:
1. Loads aggregated dataset
2. Balances classes (undersampling/oversampling)
3. Shuffles data
4. Creates stratified train/val/test splits (70/15/15)

Usage:
    python balance_dataset.py [--input ../prepared/aggregated.csv] [--ratio 1.0]
"""

import os
import sys
import csv
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetBalancer:
    """Balance and split phishing detection datasets"""

    def __init__(self, output_dir: str = "../prepared", random_seed: int = 42):
        """
        Initialize balancer

        Args:
            output_dir: Directory to save balanced data
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.random_seed = random_seed
        random.seed(random_seed)

        self.records: List[Dict] = []
        self.phishing: List[Dict] = []
        self.legitimate: List[Dict] = []

    def load_data(self, filepath: str) -> int:
        """
        Load aggregated dataset

        Args:
            filepath: Path to aggregated CSV file

        Returns:
            Number of records loaded
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return 0

        self.records = []
        self.phishing = []
        self.legitimate = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    label = row.get('label', '').lower()
                    if label == 'phishing':
                        self.phishing.append(row)
                    elif label == 'legitimate':
                        self.legitimate.append(row)
                    # Skip unknown labels

            logger.info(f"Loaded {len(self.phishing)} phishing, {len(self.legitimate)} legitimate")
            return len(self.phishing) + len(self.legitimate)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return 0

    def balance(self, ratio: float = 1.0, strategy: str = 'undersample') -> Tuple[List[Dict], List[Dict]]:
        """
        Balance the dataset

        Args:
            ratio: Target ratio of phishing:legitimate (1.0 = equal)
            strategy: 'undersample' (reduce majority) or 'oversample' (increase minority)

        Returns:
            Tuple of (balanced_phishing, balanced_legitimate)
        """
        n_phishing = len(self.phishing)
        n_legitimate = len(self.legitimate)

        if n_phishing == 0 or n_legitimate == 0:
            logger.warning("One class is empty, cannot balance")
            return self.phishing.copy(), self.legitimate.copy()

        logger.info(f"Original: {n_phishing} phishing, {n_legitimate} legitimate")
        logger.info(f"Target ratio: {ratio} (phishing:legitimate)")

        if strategy == 'undersample':
            # Reduce the larger class to match smaller * ratio
            if n_phishing > n_legitimate * ratio:
                target_phishing = int(n_legitimate * ratio)
                balanced_phishing = random.sample(self.phishing, min(target_phishing, n_phishing))
                balanced_legitimate = self.legitimate.copy()
            else:
                target_legitimate = int(n_phishing / ratio)
                balanced_phishing = self.phishing.copy()
                balanced_legitimate = random.sample(self.legitimate, min(target_legitimate, n_legitimate))

        elif strategy == 'oversample':
            # Increase the smaller class to match larger / ratio
            if n_phishing < n_legitimate * ratio:
                target_phishing = int(n_legitimate * ratio)
                # Oversample with replacement
                balanced_phishing = self.phishing.copy()
                while len(balanced_phishing) < target_phishing:
                    balanced_phishing.extend(random.choices(self.phishing,
                                                            k=min(target_phishing - len(balanced_phishing),
                                                                  len(self.phishing))))
                balanced_legitimate = self.legitimate.copy()
            else:
                target_legitimate = int(n_phishing / ratio)
                balanced_phishing = self.phishing.copy()
                balanced_legitimate = self.legitimate.copy()
                while len(balanced_legitimate) < target_legitimate:
                    balanced_legitimate.extend(random.choices(self.legitimate,
                                                               k=min(target_legitimate - len(balanced_legitimate),
                                                                     len(self.legitimate))))
        else:
            logger.warning(f"Unknown strategy: {strategy}, using undersample")
            return self.balance(ratio, 'undersample')

        logger.info(f"Balanced: {len(balanced_phishing)} phishing, {len(balanced_legitimate)} legitimate")

        return balanced_phishing, balanced_legitimate

    def create_splits(self, phishing: List[Dict], legitimate: List[Dict],
                      train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """
        Create stratified train/validation/test splits

        Args:
            phishing: Phishing records
            legitimate: Legitimate records
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.15)

        Returns:
            Dictionary with 'train', 'val', 'test' splits
        """
        test_ratio = 1.0 - train_ratio - val_ratio

        # Shuffle both classes
        random.shuffle(phishing)
        random.shuffle(legitimate)

        # Calculate split indices for phishing
        n_phishing = len(phishing)
        phishing_train_end = int(n_phishing * train_ratio)
        phishing_val_end = int(n_phishing * (train_ratio + val_ratio))

        phishing_train = phishing[:phishing_train_end]
        phishing_val = phishing[phishing_train_end:phishing_val_end]
        phishing_test = phishing[phishing_val_end:]

        # Calculate split indices for legitimate
        n_legitimate = len(legitimate)
        legitimate_train_end = int(n_legitimate * train_ratio)
        legitimate_val_end = int(n_legitimate * (train_ratio + val_ratio))

        legitimate_train = legitimate[:legitimate_train_end]
        legitimate_val = legitimate[legitimate_train_end:legitimate_val_end]
        legitimate_test = legitimate[legitimate_val_end:]

        # Combine and shuffle each split
        train = phishing_train + legitimate_train
        val = phishing_val + legitimate_val
        test = phishing_test + legitimate_test

        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        logger.info(f"Train: {len(train)} ({len(phishing_train)} phishing, {len(legitimate_train)} legitimate)")
        logger.info(f"Val: {len(val)} ({len(phishing_val)} phishing, {len(legitimate_val)} legitimate)")
        logger.info(f"Test: {len(test)} ({len(phishing_test)} phishing, {len(legitimate_test)} legitimate)")

        return {
            'train': train,
            'val': val,
            'test': test
        }

    def save_split(self, records: List[Dict], filename: str) -> str:
        """
        Save a split to CSV

        Args:
            records: Records to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        if not records:
            logger.warning(f"No records to save for {filename}")
            return str(output_path)

        fieldnames = ['text', 'label', 'source', 'content_type']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Saved {len(records)} records to {output_path}")
        return str(output_path)

    def save_all_splits(self, splits: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        Save all splits to files

        Args:
            splits: Dictionary with train/val/test splits

        Returns:
            Dictionary with file paths
        """
        paths = {}
        paths['train'] = self.save_split(splits['train'], 'train.csv')
        paths['val'] = self.save_split(splits['val'], 'validation.csv')
        paths['test'] = self.save_split(splits['test'], 'test.csv')
        return paths

    def save_stats(self, splits: Dict[str, List[Dict]], filename: str = "split_stats.txt") -> str:
        """
        Save split statistics

        Args:
            splits: Dictionary with splits
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Dataset Split Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")

            for split_name, records in splits.items():
                phishing_count = sum(1 for r in records if r.get('label') == 'phishing')
                legitimate_count = sum(1 for r in records if r.get('label') == 'legitimate')

                f.write(f"{split_name.upper()}:\n")
                f.write(f"  Total: {len(records)}\n")
                f.write(f"  Phishing: {phishing_count} ({phishing_count/len(records)*100:.1f}%)\n")
                f.write(f"  Legitimate: {legitimate_count} ({legitimate_count/len(records)*100:.1f}%)\n")

                # By content type
                by_type = defaultdict(int)
                for r in records:
                    by_type[r.get('content_type', 'unknown')] += 1
                f.write("  By content type:\n")
                for ctype, count in sorted(by_type.items()):
                    f.write(f"    {ctype}: {count}\n")
                f.write("\n")

        logger.info(f"Saved statistics to {output_path}")
        return str(output_path)

    def run(self, input_file: str, ratio: float = 1.0,
            strategy: str = 'undersample') -> Dict:
        """
        Run the full balancing and splitting pipeline

        Args:
            input_file: Path to aggregated dataset
            ratio: Target class ratio
            strategy: Balancing strategy

        Returns:
            Summary of results
        """
        logger.info("Starting dataset balancing...")
        start_time = datetime.now()

        # Load data
        total = self.load_data(input_file)
        if total == 0:
            return {
                'success': False,
                'error': 'Failed to load data'
            }

        # Balance
        balanced_phishing, balanced_legitimate = self.balance(ratio, strategy)

        # Create splits
        splits = self.create_splits(balanced_phishing, balanced_legitimate)

        # Save
        paths = self.save_all_splits(splits)
        stats_path = self.save_stats(splits)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'success': True,
            'train_count': len(splits['train']),
            'val_count': len(splits['val']),
            'test_count': len(splits['test']),
            'total': sum(len(s) for s in splits.values()),
            'files': paths,
            'stats_file': stats_path,
            'elapsed_seconds': elapsed
        }


def main():
    parser = argparse.ArgumentParser(
        description='Balance and split phishing detection dataset'
    )
    parser.add_argument(
        '--input',
        default='../prepared/aggregated.csv',
        help='Path to aggregated dataset'
    )
    parser.add_argument(
        '--output',
        default='../prepared',
        help='Output directory for splits'
    )
    parser.add_argument(
        '--ratio',
        type=float,
        default=1.0,
        help='Target phishing:legitimate ratio (default: 1.0 = equal)'
    )
    parser.add_argument(
        '--strategy',
        choices=['undersample', 'oversample'],
        default='undersample',
        help='Balancing strategy (default: undersample)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    balancer = DatasetBalancer(
        output_dir=args.output,
        random_seed=args.seed
    )

    result = balancer.run(
        input_file=args.input,
        ratio=args.ratio,
        strategy=args.strategy
    )

    if result['success']:
        print(f"\nBalancing complete!")
        print(f"Train: {result['train_count']}")
        print(f"Validation: {result['val_count']}")
        print(f"Test: {result['test_count']}")
        print(f"Total: {result['total']}")
        print(f"Stats file: {result['stats_file']}")
        print(f"Time: {result['elapsed_seconds']:.1f}s")
    else:
        print(f"\nBalancing failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
