"""
UCI SMS Spam Collection Downloader
Downloads SMS spam/ham dataset from UCI Machine Learning Repository

Dataset Information:
- 5,574 SMS messages
- 747 spam, 4,827 ham (legitimate)
- Tagged as smishing/spam or legitimate
- Widely used baseline for SMS spam detection

Usage:
    python download_uci_sms.py [--output uci_sms_spam.csv]

Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
"""

import os
import sys
import csv
import logging
import argparse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from io import BytesIO

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# UCI SMS Spam Collection URL
UCI_SMS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"


class UCISMSDownloader:
    """Download and process UCI SMS Spam Collection"""

    def __init__(self, output_dir: str = "."):
        """
        Initialize downloader

        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'uci-sms-downloader/ai-anti-spam-shield'
        })

    def download_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Download the UCI SMS Spam Collection

        Returns:
            Tuple of (spam_messages, ham_messages)
        """
        logger.info("Downloading UCI SMS Spam Collection...")

        try:
            # Download the ZIP file
            response = self.session.get(UCI_SMS_URL, timeout=60)
            response.raise_for_status()

            # Extract from ZIP
            logger.info("Extracting data...")
            spam_messages = []
            ham_messages = []

            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # Find the data file
                for filename in z.namelist():
                    if 'SMSSpamCollection' in filename or filename.endswith('.txt'):
                        with z.open(filename) as f:
                            content = f.read().decode('utf-8', errors='ignore')
                            break
                else:
                    # Try reading any file
                    with z.open(z.namelist()[0]) as f:
                        content = f.read().decode('utf-8', errors='ignore')

            # Parse the tab-separated file (format: label\tmessage)
            for line in content.splitlines():
                line = line.strip()
                if not line or '\t' not in line:
                    continue

                parts = line.split('\t', 1)
                if len(parts) != 2:
                    continue

                label, message = parts
                label = label.strip().lower()
                message = message.strip()

                if not message:
                    continue

                record = {
                    'text': message,
                    'original_label': label,
                    'source': 'uci_sms',
                    'type': 'sms'
                }

                if label == 'spam':
                    record['label'] = 'smishing'  # Label as smishing for our dataset
                    spam_messages.append(record)
                elif label == 'ham':
                    record['label'] = 'legitimate'
                    ham_messages.append(record)

            logger.info(f"Downloaded {len(spam_messages)} spam and {len(ham_messages)} ham messages")
            return spam_messages, ham_messages

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download UCI SMS dataset: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Failed to process UCI SMS data: {e}")
            return [], []

    def save_to_csv(self, spam: List[Dict], ham: List[Dict],
                    filename: str = "uci_sms_collection.csv") -> str:
        """
        Save all messages to CSV file

        Args:
            spam: Spam messages
            ham: Ham messages
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        all_records = spam + ham
        if not all_records:
            logger.warning("No records to save")
            return str(output_path)

        fieldnames = ['text', 'label', 'original_label', 'source', 'type']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

        logger.info(f"Saved {len(all_records)} records to {output_path}")
        return str(output_path)

    def save_spam_only(self, spam: List[Dict], filename: str = "uci_sms_spam_only.csv") -> str:
        """
        Save only spam/smishing messages (for phishing training)

        Args:
            spam: Spam messages
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            for record in spam:
                writer.writerow([record['text'], 'smishing'])

        logger.info(f"Saved {len(spam)} spam messages to {output_path}")
        return str(output_path)

    def save_ham_only(self, ham: List[Dict], filename: str = "uci_sms_ham_only.csv") -> str:
        """
        Save only ham/legitimate messages (for balance)

        Args:
            ham: Ham messages
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            for record in ham:
                writer.writerow([record['text'], 'legitimate'])

        logger.info(f"Saved {len(ham)} ham messages to {output_path}")
        return str(output_path)

    def run(self) -> Dict:
        """
        Run the full download pipeline

        Returns:
            Summary of download results
        """
        logger.info("Starting UCI SMS download...")
        start_time = datetime.now()

        # Download dataset
        spam, ham = self.download_dataset()

        if not spam and not ham:
            return {
                'success': False,
                'error': 'Failed to download dataset',
                'spam_count': 0,
                'ham_count': 0
            }

        # Save to files
        full_path = self.save_to_csv(spam, ham)
        spam_path = self.save_spam_only(spam)
        ham_path = self.save_ham_only(ham)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'success': True,
            'spam_count': len(spam),
            'ham_count': len(ham),
            'total': len(spam) + len(ham),
            'full_file': full_path,
            'spam_file': spam_path,
            'ham_file': ham_path,
            'elapsed_seconds': elapsed
        }


def main():
    parser = argparse.ArgumentParser(
        description='Download UCI SMS Spam Collection dataset'
    )
    parser.add_argument(
        '--output',
        default='.',
        help='Output directory (default: current directory)'
    )

    args = parser.parse_args()

    downloader = UCISMSDownloader(output_dir=args.output)
    result = downloader.run()

    if result['success']:
        print(f"\nDownload complete!")
        print(f"Spam messages: {result['spam_count']}")
        print(f"Ham messages: {result['ham_count']}")
        print(f"Total: {result['total']}")
        print(f"Full file: {result['full_file']}")
        print(f"Spam file: {result['spam_file']}")
        print(f"Ham file: {result['ham_file']}")
        print(f"Time: {result['elapsed_seconds']:.1f}s")
    else:
        print(f"\nDownload failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
