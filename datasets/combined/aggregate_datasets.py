"""
Dataset Aggregator
Combines all phishing and legitimate datasets into a unified format

This script:
1. Loads data from PhishTank, OpenPhish, UCI SMS, and Tranco
2. Normalizes labels and formats
3. Removes duplicates
4. Creates a unified dataset for training

Usage:
    python aggregate_datasets.py [--output ../prepared/aggregated.csv]
"""

import os
import sys
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetAggregator:
    """Aggregate multiple phishing and legitimate datasets"""

    # Standard label mapping
    LABEL_MAP = {
        # Phishing labels
        'phishing': 'phishing',
        'spam': 'phishing',
        'smishing': 'phishing',
        'scam': 'phishing',
        'malicious': 'phishing',

        # Legitimate labels
        'legitimate': 'legitimate',
        'ham': 'legitimate',
        'safe': 'legitimate',
        'benign': 'legitimate',
    }

    # Source type mapping
    SOURCE_TYPE_MAP = {
        'phishtank': 'url',
        'openphish': 'url',
        'tranco': 'url',
        'uci_sms': 'sms',
        'enron': 'email',
        'kaggle': 'mixed',
    }

    def __init__(self, datasets_dir: str = ".", output_dir: str = "../prepared"):
        """
        Initialize aggregator

        Args:
            datasets_dir: Root directory containing dataset subdirectories
            output_dir: Directory to save aggregated data
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.records: List[Dict] = []
        self.seen_hashes: Set[str] = set()
        self.stats = {
            'phishing': 0,
            'legitimate': 0,
            'duplicates_removed': 0,
            'sources': {}
        }

    def _hash_text(self, text: str) -> str:
        """Create hash of text for deduplication"""
        normalized = text.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def _normalize_label(self, label: str) -> str:
        """Normalize label to standard format"""
        label_lower = label.lower().strip()
        return self.LABEL_MAP.get(label_lower, 'unknown')

    def _add_record(self, text: str, label: str, source: str,
                    content_type: str = 'text') -> bool:
        """
        Add a record if not duplicate

        Args:
            text: Text content (URL or message)
            label: Original label
            source: Data source name
            content_type: Type of content (url, sms, email)

        Returns:
            True if record was added, False if duplicate
        """
        if not text or not text.strip():
            return False

        text = text.strip()
        text_hash = self._hash_text(text)

        # Skip duplicates
        if text_hash in self.seen_hashes:
            self.stats['duplicates_removed'] += 1
            return False

        self.seen_hashes.add(text_hash)

        normalized_label = self._normalize_label(label)
        if normalized_label == 'unknown':
            logger.warning(f"Unknown label '{label}' from {source}")
            return False

        record = {
            'text': text,
            'label': normalized_label,
            'source': source,
            'content_type': content_type,
            'text_hash': text_hash
        }

        self.records.append(record)

        # Update stats
        self.stats[normalized_label] += 1
        if source not in self.stats['sources']:
            self.stats['sources'][source] = {'phishing': 0, 'legitimate': 0}
        self.stats['sources'][source][normalized_label] += 1

        return True

    def load_csv_file(self, filepath: Path, text_col: str = 'text',
                      label_col: str = 'label', source: str = 'unknown',
                      content_type: str = 'text') -> int:
        """
        Load records from a CSV file

        Args:
            filepath: Path to CSV file
            text_col: Name of text column
            label_col: Name of label column
            source: Source name for records
            content_type: Type of content

        Returns:
            Number of records loaded
        """
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return 0

        count = 0
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)

                # Handle different column name cases
                fieldnames_lower = {fn.lower(): fn for fn in reader.fieldnames or []}

                text_col_actual = fieldnames_lower.get(text_col.lower(), text_col)
                label_col_actual = fieldnames_lower.get(label_col.lower(), label_col)

                for row in reader:
                    text = row.get(text_col_actual, row.get('text', row.get('url', '')))
                    label = row.get(label_col_actual, row.get('label', ''))

                    if self._add_record(text, label, source, content_type):
                        count += 1

            logger.info(f"Loaded {count} records from {filepath}")

        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

        return count

    def load_phishtank(self) -> int:
        """Load PhishTank phishing URLs"""
        filepath = self.datasets_dir / 'phishing' / 'phishtank_urls_simple.csv'
        return self.load_csv_file(filepath, source='phishtank', content_type='url')

    def load_openphish(self) -> int:
        """Load OpenPhish phishing URLs"""
        filepath = self.datasets_dir / 'phishing' / 'openphish_urls_simple.csv'
        return self.load_csv_file(filepath, source='openphish', content_type='url')

    def load_tranco(self) -> int:
        """Load Tranco legitimate URLs"""
        filepath = self.datasets_dir / 'legitimate' / 'tranco_urls_simple.csv'
        return self.load_csv_file(filepath, source='tranco', content_type='url')

    def load_uci_sms(self) -> int:
        """Load UCI SMS spam collection"""
        filepath = self.datasets_dir / 'smishing' / 'uci_sms_collection.csv'
        return self.load_csv_file(filepath, source='uci_sms', content_type='sms')

    def load_all(self) -> int:
        """
        Load all available datasets

        Returns:
            Total number of records loaded
        """
        logger.info("Loading all datasets...")

        total = 0
        total += self.load_phishtank()
        total += self.load_openphish()
        total += self.load_tranco()
        total += self.load_uci_sms()

        logger.info(f"Total records loaded: {total}")
        logger.info(f"Phishing: {self.stats['phishing']}, Legitimate: {self.stats['legitimate']}")
        logger.info(f"Duplicates removed: {self.stats['duplicates_removed']}")

        return total

    def save_aggregated(self, filename: str = "aggregated.csv") -> str:
        """
        Save aggregated dataset to CSV

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        if not self.records:
            logger.warning("No records to save")
            return str(output_path)

        fieldnames = ['text', 'label', 'source', 'content_type']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.records)

        logger.info(f"Saved {len(self.records)} records to {output_path}")
        return str(output_path)

    def save_stats(self, filename: str = "aggregation_stats.txt") -> str:
        """
        Save aggregation statistics

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Dataset Aggregation Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            f.write("Overall Statistics:\n")
            f.write(f"  Total records: {len(self.records)}\n")
            f.write(f"  Phishing: {self.stats['phishing']}\n")
            f.write(f"  Legitimate: {self.stats['legitimate']}\n")
            f.write(f"  Duplicates removed: {self.stats['duplicates_removed']}\n\n")

            f.write("By Source:\n")
            for source, counts in self.stats['sources'].items():
                f.write(f"  {source}:\n")
                f.write(f"    Phishing: {counts['phishing']}\n")
                f.write(f"    Legitimate: {counts['legitimate']}\n")

        logger.info(f"Saved statistics to {output_path}")
        return str(output_path)

    def run(self) -> Dict:
        """
        Run the full aggregation pipeline

        Returns:
            Summary of aggregation results
        """
        logger.info("Starting dataset aggregation...")
        start_time = datetime.now()

        # Load all datasets
        self.load_all()

        # Save results
        data_path = self.save_aggregated()
        stats_path = self.save_stats()

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'success': True,
            'total_records': len(self.records),
            'phishing': self.stats['phishing'],
            'legitimate': self.stats['legitimate'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'data_file': data_path,
            'stats_file': stats_path,
            'elapsed_seconds': elapsed
        }


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate multiple phishing/legitimate datasets'
    )
    parser.add_argument(
        '--datasets-dir',
        default='..',
        help='Root directory containing dataset subdirectories'
    )
    parser.add_argument(
        '--output',
        default='../prepared',
        help='Output directory for aggregated data'
    )

    args = parser.parse_args()

    aggregator = DatasetAggregator(
        datasets_dir=args.datasets_dir,
        output_dir=args.output
    )

    result = aggregator.run()

    if result['success']:
        print(f"\nAggregation complete!")
        print(f"Total records: {result['total_records']}")
        print(f"Phishing: {result['phishing']}")
        print(f"Legitimate: {result['legitimate']}")
        print(f"Duplicates removed: {result['duplicates_removed']}")
        print(f"Data file: {result['data_file']}")
        print(f"Stats file: {result['stats_file']}")
        print(f"Time: {result['elapsed_seconds']:.1f}s")
    else:
        print(f"\nAggregation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
