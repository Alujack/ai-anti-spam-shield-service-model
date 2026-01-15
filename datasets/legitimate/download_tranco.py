"""
Tranco Top Sites Downloader
Downloads legitimate URLs from Tranco top 1M sites list

Tranco provides:
- Research-oriented top sites list
- Combination of Alexa, Umbrella, Majestic, and Quantcast
- More stable than individual lists
- Perfect for balancing phishing datasets with legitimate URLs

Usage:
    python download_tranco.py [--output tranco_top_1m.csv] [--limit 100000]

Website: https://tranco-list.eu/
"""

import os
import sys
import csv
import logging
import argparse
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict
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

# Tranco list URL (latest list)
TRANCO_LIST_URL = "https://tranco-list.eu/top-1m.csv.zip"
TRANCO_API_URL = "https://tranco-list.eu/api/lists/date/latest"


class TrancoDownloader:
    """Download and process Tranco top sites list"""

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
            'User-Agent': 'tranco-downloader/ai-anti-spam-shield'
        })

    def get_latest_list_id(self) -> str:
        """
        Get the ID of the latest Tranco list

        Returns:
            List ID string
        """
        try:
            response = self.session.get(TRANCO_API_URL, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('list_id', '')
        except Exception as e:
            logger.warning(f"Failed to get latest list ID: {e}")
            return ''

    def download_list(self, limit: int = 100000) -> List[Dict]:
        """
        Download the Tranco top sites list

        Args:
            limit: Maximum number of sites to download

        Returns:
            List of site records
        """
        logger.info("Downloading Tranco top sites list...")

        try:
            # Download the ZIP file
            response = self.session.get(TRANCO_LIST_URL, timeout=120)
            response.raise_for_status()

            # Extract CSV from ZIP
            logger.info("Extracting data...")
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                # Find the CSV file in the archive
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV file found in archive")

                with z.open(csv_files[0]) as f:
                    content = f.read().decode('utf-8')

            # Parse CSV (format: rank,domain)
            sites = []
            for line in content.splitlines():
                if not line.strip():
                    continue

                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        rank = int(parts[0])
                        domain = parts[1].strip()

                        if domain and rank <= limit:
                            sites.append({
                                'rank': rank,
                                'domain': domain,
                                'url': f'https://{domain}',
                                'source': 'tranco',
                                'label': 'legitimate'
                            })
                    except ValueError:
                        continue

                if len(sites) >= limit:
                    break

            logger.info(f"Downloaded {len(sites)} sites from Tranco")
            return sites

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download Tranco list: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to process Tranco data: {e}")
            return []

    def save_to_csv(self, records: List[Dict], filename: str = "tranco_top_sites.csv") -> str:
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

        fieldnames = ['rank', 'domain', 'url', 'source', 'label']

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        logger.info(f"Saved {len(records)} records to {output_path}")
        return str(output_path)

    def save_urls_only(self, records: List[Dict], filename: str = "tranco_urls_simple.csv") -> str:
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
                writer.writerow([record['url'], 'legitimate'])

        logger.info(f"Saved {len(records)} URLs to {output_path}")
        return str(output_path)

    def save_domains_only(self, records: List[Dict], filename: str = "tranco_domains.txt") -> str:
        """
        Save only domains (for whitelist purposes)

        Args:
            records: Processed records
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(f"{record['domain']}\n")

        logger.info(f"Saved {len(records)} domains to {output_path}")
        return str(output_path)

    def run(self, limit: int = 100000) -> Dict:
        """
        Run the full download pipeline

        Args:
            limit: Maximum sites to download

        Returns:
            Summary of download results
        """
        logger.info("Starting Tranco download...")
        start_time = datetime.now()

        # Get latest list info
        list_id = self.get_latest_list_id()
        if list_id:
            logger.info(f"Using Tranco list: {list_id}")

        # Download list
        sites = self.download_list(limit)

        if not sites:
            return {
                'success': False,
                'error': 'Failed to download list',
                'records': 0
            }

        # Save to files
        full_path = self.save_to_csv(sites)
        simple_path = self.save_urls_only(sites)
        domains_path = self.save_domains_only(sites)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            'success': True,
            'records': len(sites),
            'list_id': list_id,
            'full_file': full_path,
            'simple_file': simple_path,
            'domains_file': domains_path,
            'elapsed_seconds': elapsed
        }


def main():
    parser = argparse.ArgumentParser(
        description='Download Tranco top sites list (legitimate URLs)'
    )
    parser.add_argument(
        '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100000,
        help='Maximum sites to download (default: 100000)'
    )

    args = parser.parse_args()

    downloader = TrancoDownloader(output_dir=args.output)
    result = downloader.run(limit=args.limit)

    if result['success']:
        print(f"\nDownload complete!")
        print(f"Records: {result['records']}")
        if result.get('list_id'):
            print(f"List ID: {result['list_id']}")
        print(f"Full file: {result['full_file']}")
        print(f"Simple file: {result['simple_file']}")
        print(f"Domains file: {result['domains_file']}")
        print(f"Time: {result['elapsed_seconds']:.1f}s")
    else:
        print(f"\nDownload failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
