"""
PhishTank Dataset Downloader
Downloads verified phishing URLs from PhishTank database

PhishTank provides:
- Verified phishing URLs submitted by community
- Regular updates with new phishing sites
- API and bulk download options

Usage:
    python download_phishtank.py [--api-key YOUR_KEY] [--output phishtank_urls.csv]

Note:
    - Free API key available at: https://phishtank.org/api_info.php
    - Rate limit: 1000 requests/day with API key
    - Bulk download available without API key
"""

import os
import sys
import csv
import json
import gzip
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
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

# PhishTank URLs
PHISHTANK_DB_URL = "http://data.phishtank.com/data/online-valid.json.gz"
PHISHTANK_API_URL = "https://checkurl.phishtank.com/checkurl/"


class PhishTankDownloader:
    """Download and process PhishTank phishing URL database"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "."):
        """
        Initialize downloader

        Args:
            api_key: Optional PhishTank API key (for higher rate limits)
            output_dir: Directory to save downloaded data
        """
        self.api_key = api_key or os.environ.get('PHISHTANK_API_KEY')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'phishtank/ai-anti-spam-shield'
        })

    def download_database(self, max_urls: int = 50000) -> List[Dict]:
        """
        Download the full PhishTank database (bulk download)

        Args:
            max_urls: Maximum number of URLs to download

        Returns:
            List of phishing URL records
        """
        logger.info("Downloading PhishTank database (this may take a while)...")

        try:
            # Download compressed database
            response = self.session.get(PHISHTANK_DB_URL, stream=True, timeout=120)
            response.raise_for_status()

            # Decompress and parse JSON
            logger.info("Decompressing database...")
            compressed_data = BytesIO(response.content)

            with gzip.GzipFile(fileobj=compressed_data) as f:
                data = json.loads(f.read().decode('utf-8'))

            logger.info(f"Downloaded {len(data)} phishing URLs from PhishTank")

            # Limit to max_urls
            if len(data) > max_urls:
                logger.info(f"Limiting to {max_urls} URLs")
                data = data[:max_urls]

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download PhishTank database: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse PhishTank data: {e}")
            return []

    def process_records(self, records: List[Dict]) -> List[Dict]:
        """
        Process raw PhishTank records into standardized format

        Args:
            records: Raw PhishTank records

        Returns:
            Processed records with standardized fields
        """
        processed = []

        for record in records:
            try:
                processed_record = {
                    'url': record.get('url', ''),
                    'phish_id': record.get('phish_id', ''),
                    'phish_detail_url': record.get('phish_detail_url', ''),
                    'submission_time': record.get('submission_time', ''),
                    'verified': record.get('verified', 'no'),
                    'verified_time': record.get('verification_time', ''),
                    'online': record.get('online', 'no'),
                    'target': record.get('target', ''),
                    'source': 'phishtank',
                    'label': 'phishing'
                }

                # Only include verified phishing URLs
                if processed_record['verified'] == 'yes' and processed_record['url']:
                    processed.append(processed_record)

            except Exception as e:
                logger.warning(f"Failed to process record: {e}")
                continue

        logger.info(f"Processed {len(processed)} verified phishing URLs")
        return processed

    def save_to_csv(self, records: List[Dict], filename: str = "phishtank_urls.csv") -> str:
        """
        Save processed records to CSV file

        Args:
            records: Processed records to save
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        if not records:
            logger.warning("No records to save")
            return str(output_path)

        fieldnames = ['url', 'phish_id', 'target', 'submission_time',
                      'verified_time', 'source', 'label']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Saved {len(records)} records to {output_path}")
        return str(output_path)

    def save_urls_only(self, records: List[Dict], filename: str = "phishtank_urls_simple.csv") -> str:
        """
        Save only URLs and labels (simplified format for ML training)

        Args:
            records: Processed records
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['text', 'label'])
            for record in records:
                writer.writerow([record['url'], 'phishing'])

        logger.info(f"Saved {len(records)} URLs to {output_path}")
        return str(output_path)

    def check_url(self, url: str) -> Dict:
        """
        Check a single URL against PhishTank API

        Args:
            url: URL to check

        Returns:
            API response with phishing status
        """
        if not self.api_key:
            logger.warning("API key required for URL checking")
            return {'error': 'API key required'}

        try:
            data = {
                'url': url,
                'format': 'json',
                'app_key': self.api_key
            }

            response = self.session.post(PHISHTANK_API_URL, data=data, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"API check failed: {e}")
            return {'error': str(e)}

    def run(self, max_urls: int = 50000) -> Dict:
        """
        Run the full download pipeline

        Args:
            max_urls: Maximum URLs to download

        Returns:
            Summary of download results
        """
        logger.info("Starting PhishTank download...")
        start_time = datetime.now()

        # Download database
        raw_records = self.download_database(max_urls)

        if not raw_records:
            return {
                'success': False,
                'error': 'Failed to download database',
                'records': 0
            }

        # Process records
        processed = self.process_records(raw_records)

        # Save to files
        full_path = self.save_to_csv(processed)
        simple_path = self.save_urls_only(processed)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'success': True,
            'records': len(processed),
            'full_file': full_path,
            'simple_file': simple_path,
            'elapsed_seconds': elapsed
        }


def main():
    parser = argparse.ArgumentParser(
        description='Download PhishTank phishing URL database'
    )
    parser.add_argument(
        '--api-key',
        help='PhishTank API key (optional, can also use PHISHTANK_API_KEY env var)'
    )
    parser.add_argument(
        '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--max-urls',
        type=int,
        default=50000,
        help='Maximum URLs to download (default: 50000)'
    )

    args = parser.parse_args()

    downloader = PhishTankDownloader(
        api_key=args.api_key,
        output_dir=args.output
    )

    result = downloader.run(max_urls=args.max_urls)

    if result['success']:
        print(f"\nDownload complete!")
        print(f"Records: {result['records']}")
        print(f"Full file: {result['full_file']}")
        print(f"Simple file: {result['simple_file']}")
        print(f"Time: {result['elapsed_seconds']:.1f}s")
    else:
        print(f"\nDownload failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
