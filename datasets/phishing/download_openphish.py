"""
OpenPhish Dataset Downloader
Downloads phishing URLs from OpenPhish community feed

OpenPhish provides:
- Community-curated phishing URLs
- Free public feed (no API key required)
- Regular updates

Usage:
    python download_openphish.py [--output openphish_urls.csv]

Feed URL: https://openphish.com/feed.txt
"""

import os
import sys
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse

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

# OpenPhish feed URL
OPENPHISH_FEED_URL = "https://openphish.com/feed.txt"


class OpenPhishDownloader:
    """Download and process OpenPhish phishing URL feed"""

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
            'User-Agent': 'openphish-downloader/ai-anti-spam-shield'
        })

    def download_feed(self) -> List[str]:
        """
        Download the OpenPhish feed

        Returns:
            List of phishing URLs
        """
        logger.info("Downloading OpenPhish feed...")

        try:
            response = self.session.get(OPENPHISH_FEED_URL, timeout=60)
            response.raise_for_status()

            # Parse URLs (one per line)
            urls = [
                line.strip()
                for line in response.text.splitlines()
                if line.strip() and line.strip().startswith('http')
            ]

            logger.info(f"Downloaded {len(urls)} phishing URLs from OpenPhish")
            return urls

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download OpenPhish feed: {e}")
            return []

    def extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return ''

    def process_urls(self, urls: List[str]) -> List[Dict]:
        """
        Process URLs into standardized format

        Args:
            urls: List of raw URLs

        Returns:
            List of processed records
        """
        processed = []
        seen_urls = set()

        for url in urls:
            # Skip duplicates
            if url in seen_urls:
                continue
            seen_urls.add(url)

            try:
                record = {
                    'url': url,
                    'domain': self.extract_domain(url),
                    'source': 'openphish',
                    'label': 'phishing',
                    'download_time': datetime.now().isoformat()
                }
                processed.append(record)

            except Exception as e:
                logger.warning(f"Failed to process URL {url}: {e}")
                continue

        logger.info(f"Processed {len(processed)} unique URLs")
        return processed

    def save_to_csv(self, records: List[Dict], filename: str = "openphish_urls.csv") -> str:
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

        fieldnames = ['url', 'domain', 'source', 'label', 'download_time']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Saved {len(records)} records to {output_path}")
        return str(output_path)

    def save_urls_only(self, records: List[Dict], filename: str = "openphish_urls_simple.csv") -> str:
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

    def run(self) -> Dict:
        """
        Run the full download pipeline

        Returns:
            Summary of download results
        """
        logger.info("Starting OpenPhish download...")
        start_time = datetime.now()

        # Download feed
        urls = self.download_feed()

        if not urls:
            return {
                'success': False,
                'error': 'Failed to download feed',
                'records': 0
            }

        # Process URLs
        processed = self.process_urls(urls)

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
        description='Download OpenPhish phishing URL feed'
    )
    parser.add_argument(
        '--output',
        default='.',
        help='Output directory (default: current directory)'
    )

    args = parser.parse_args()

    downloader = OpenPhishDownloader(output_dir=args.output)
    result = downloader.run()

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
