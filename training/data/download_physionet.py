"""
PhysioNet Challenge 2019 Data Downloader.

Dataset Information:
- Name: PhysioNet Computing in Cardiology Challenge 2019
- URL: https://physionet.org/content/challenge-2019/1.0.0/
- License: Open Database License (ODbL) v1.0
- Cost: FREE (no payment, no registration required)
- Size: ~200MB extracted (40,336 patient files)
- Patients: 40,336 total (20,336 in setA, 20,000 in setB)
- Sepsis Rate: ~7.3% of patients, ~2.7% of hourly records

Usage:
    python -m training.data.download_physionet --output-dir data/physionet

The script will:
1. Download patient PSV files from PhysioNet
2. Save to training_setA/ and training_setB/ directories
3. Generate a metadata summary
"""

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# PhysioNet Challenge 2019 URLs (public access, no authentication required)
BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0"
TRAINING_SETS = {
    "training_setA": {
        "url": f"{BASE_URL}/training/training_setA/",
        "expected_count": 20336,
        "description": "Training Set A - ICU data from hospital system 1",
    },
    "training_setB": {
        "url": f"{BASE_URL}/training/training_setB/",
        "expected_count": 20000,
        "description": "Training Set B - ICU data from hospital system 2",
    },
}


def get_file_list(set_url: str) -> List[str]:
    """
    Get list of PSV files from a PhysioNet directory.

    Args:
        set_url: URL to the training set directory

    Returns:
        List of PSV filenames
    """
    try:
        response = requests.get(set_url, timeout=30)
        response.raise_for_status()
        files = re.findall(r'href="(p\d+\.psv)"', response.text)
        return files
    except Exception as e:
        logger.error(f"Failed to get file list from {set_url}: {e}")
        return []


def download_file(args: Tuple[str, Path]) -> Tuple[str, bool, str]:
    """
    Download a single file.

    Args:
        args: Tuple of (url, output_path)

    Returns:
        Tuple of (filename, success, error_message)
    """
    url, output_path = args
    filename = output_path.name

    try:
        if output_path.exists() and output_path.stat().st_size > 0:
            return (filename, True, "already exists")

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        return (filename, True, "")

    except Exception as e:
        return (filename, False, str(e))


def download_training_set(
    set_name: str,
    output_dir: Path,
    max_workers: int = 10,
    max_files: Optional[int] = None,
) -> dict:
    """
    Download a complete training set.

    Args:
        set_name: Name of training set ("training_setA" or "training_setB")
        output_dir: Base output directory
        max_workers: Number of parallel download threads
        max_files: Maximum files to download (for testing)

    Returns:
        Dictionary with download statistics
    """
    set_info = TRAINING_SETS[set_name]
    set_url = set_info["url"]
    set_dir = output_dir / set_name

    logger.info(f"Downloading {set_info['description']}...")
    logger.info(f"  URL: {set_url}")
    logger.info(f"  Output: {set_dir}")

    # Get file list
    files = get_file_list(set_url)
    if not files:
        return {"success": False, "error": "Failed to get file list"}

    logger.info(f"  Found {len(files)} files")

    if max_files:
        files = files[:max_files]
        logger.info(f"  Limited to {len(files)} files")

    # Prepare download tasks
    tasks = []
    for filename in files:
        url = set_url + filename
        output_path = set_dir / filename
        tasks.append((url, output_path))

    # Download with progress bar
    results = {"downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, task): task for task in tasks}

        with tqdm(total=len(tasks), desc=set_name, unit="files") as pbar:
            for future in as_completed(futures):
                filename, success, message = future.result()
                if success:
                    if message == "already exists":
                        results["skipped"] += 1
                    else:
                        results["downloaded"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{filename}: {message}")
                pbar.update(1)

    logger.info(f"  Downloaded: {results['downloaded']}")
    logger.info(f"  Skipped (existing): {results['skipped']}")
    logger.info(f"  Failed: {results['failed']}")

    results["success"] = results["failed"] == 0
    results["total"] = results["downloaded"] + results["skipped"]
    return results


def verify_download(output_dir: Path) -> dict:
    """
    Verify downloaded data integrity.

    Args:
        output_dir: Directory containing extracted data

    Returns:
        Dictionary with verification results
    """
    logger.info("Verifying downloaded data...")

    results = {
        "setA_count": 0,
        "setB_count": 0,
        "total_patients": 0,
        "sepsis_count": 0,
        "valid": True,
        "errors": [],
    }

    for set_name in ["training_setA", "training_setB"]:
        set_dir = output_dir / set_name
        if set_dir.exists():
            psv_files = list(set_dir.glob("*.psv"))
            count = len(psv_files)
            if set_name == "training_setA":
                results["setA_count"] = count
            else:
                results["setB_count"] = count

            # Sample first file to verify format
            if psv_files:
                try:
                    with open(psv_files[0]) as f:
                        header = f.readline().strip()
                        if "SepsisLabel" not in header:
                            results["errors"].append(
                                f"{set_name}: Missing SepsisLabel column"
                            )
                            results["valid"] = False
                except Exception as e:
                    results["errors"].append(f"{set_name}: Cannot read file - {e}")
                    results["valid"] = False
        else:
            results["errors"].append(f"{set_name} directory not found")

    results["total_patients"] = results["setA_count"] + results["setB_count"]

    logger.info(f"  Set A patients: {results['setA_count']}")
    logger.info(f"  Set B patients: {results['setB_count']}")
    logger.info(f"  Total patients: {results['total_patients']}")

    if results["valid"]:
        logger.info("  Verification: PASSED")
    else:
        logger.error(f"  Verification: FAILED - {results['errors']}")

    return results


def download_physionet_2019(
    output_dir: Path,
    sets: Optional[List[str]] = None,
    max_workers: int = 10,
    max_files: Optional[int] = None,
    force: bool = False,
) -> dict:
    """
    Download PhysioNet Challenge 2019 dataset.

    This dataset is FREE and publicly available. No registration or
    payment is required.

    Args:
        output_dir: Directory to save data
        sets: Which sets to download ("A", "B", or both). Default: both
        max_workers: Parallel download threads
        max_files: Limit files per set (for testing)
        force: Re-download even if files exist

    Returns:
        Dictionary with download results and statistics

    License:
        Open Database License (ODbL) v1.0
        https://opendatacommons.org/licenses/odbl/1-0/
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sets is None:
        sets = ["A", "B"]

    logger.info("=" * 60)
    logger.info("PhysioNet Challenge 2019 Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Sets to download: {sets}")
    logger.info(f"Parallel workers: {max_workers}")
    logger.info("")
    logger.info("License: Open Database License (ODbL) v1.0")
    logger.info("This dataset is FREE - no payment required.")
    logger.info("=" * 60)

    results = {
        "success": True,
        "sets": {},
        "total_patients": 0,
    }

    for set_letter in sets:
        set_name = f"training_set{set_letter}"
        if set_name not in TRAINING_SETS:
            logger.warning(f"Unknown set: {set_letter}")
            continue

        set_dir = output_dir / set_name

        # Check if already downloaded
        if set_dir.exists() and not force:
            existing = len(list(set_dir.glob("*.psv")))
            expected = TRAINING_SETS[set_name]["expected_count"]
            if max_files:
                expected = min(expected, max_files)
            if existing >= expected:
                logger.info(f"Set {set_letter} already exists ({existing} files). Skipping.")
                results["sets"][set_name] = {
                    "success": True,
                    "total": existing,
                    "skipped": existing,
                    "downloaded": 0,
                }
                results["total_patients"] += existing
                continue

        # Download
        set_results = download_training_set(
            set_name=set_name,
            output_dir=output_dir,
            max_workers=max_workers,
            max_files=max_files,
        )
        results["sets"][set_name] = set_results
        results["total_patients"] += set_results.get("total", 0)

        if not set_results["success"]:
            results["success"] = False

    # Verify
    if results["success"]:
        verification = verify_download(output_dir)
        results["verification"] = verification
        if not verification["valid"]:
            results["success"] = False

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    for set_name, set_results in results["sets"].items():
        status = "OK" if set_results["success"] else "FAILED"
        logger.info(f"  {set_name}: {set_results.get('total', 0)} files [{status}]")
    logger.info(f"Total patients: {results['total_patients']}")
    logger.info(f"Success: {results['success']}")
    logger.info("=" * 60)

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download PhysioNet Challenge 2019 dataset (FREE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download both sets to data/physionet
    python -m training.data.download_physionet --output-dir data/physionet

    # Download only set A
    python -m training.data.download_physionet --output-dir data/physionet --sets A

    # Download 100 files for testing
    python -m training.data.download_physionet --output-dir data/physionet --max-files 100

    # Force re-download
    python -m training.data.download_physionet --output-dir data/physionet --force

License:
    Open Database License (ODbL) v1.0
    This dataset is FREE - no registration or payment required.
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/physionet"),
        help="Output directory for data (default: data/physionet)",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=["A", "B"],
        default=["A", "B"],
        help="Which sets to download (default: both)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        help="Maximum files per set (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel download threads (default: 10)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    results = download_physionet_2019(
        output_dir=args.output_dir,
        sets=args.sets,
        max_workers=args.workers,
        max_files=args.max_files,
        force=args.force,
    )

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
