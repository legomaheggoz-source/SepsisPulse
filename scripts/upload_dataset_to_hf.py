#!/usr/bin/env python3
"""
Upload PhysioNet Sepsis 2019 data to HuggingFace Datasets.

This script packages the local PhysioNet PSV files into a HuggingFace Dataset
for easy loading in the SepsisPulse app.

Usage:
    python scripts/upload_dataset_to_hf.py --token YOUR_HF_TOKEN

The dataset will be uploaded to: huggingface.co/datasets/legomaheggo/physionet-sepsis-2019
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


def load_psv_file(file_path: Path) -> Dict:
    """Load a single PSV file and return as a dictionary."""
    df = pd.read_csv(file_path, sep="|")

    # Get patient ID from filename
    patient_id = file_path.stem

    # Check for sepsis
    has_sepsis = df["SepsisLabel"].max() > 0 if "SepsisLabel" in df.columns else False
    sepsis_onset_hour = None
    if has_sepsis:
        sepsis_onset_hour = df[df["SepsisLabel"] == 1].index[0]

    # Get demographics (first row values that are constant)
    age = float(df["Age"].iloc[0]) if "Age" in df.columns else None
    gender = int(df["Gender"].iloc[0]) if "Gender" in df.columns else None

    return {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "icu_hours": len(df),
        "has_sepsis": has_sepsis,
        "sepsis_onset_hour": sepsis_onset_hour,
        # Store vitals as lists
        "HR": df["HR"].tolist() if "HR" in df.columns else [],
        "SBP": df["SBP"].tolist() if "SBP" in df.columns else [],
        "DBP": df["DBP"].tolist() if "DBP" in df.columns else [],
        "MAP": df["MAP"].tolist() if "MAP" in df.columns else [],
        "Resp": df["Resp"].tolist() if "Resp" in df.columns else [],
        "Temp": df["Temp"].tolist() if "Temp" in df.columns else [],
        "O2Sat": df["O2Sat"].tolist() if "O2Sat" in df.columns else [],
        "SepsisLabel": df["SepsisLabel"].tolist() if "SepsisLabel" in df.columns else [],
    }


def main():
    parser = argparse.ArgumentParser(description="Upload PhysioNet data to HuggingFace")
    parser.add_argument("--token", type=str, required=True, help="HuggingFace API token")
    parser.add_argument("--data-dir", type=str, default="data/physionet", help="Path to PhysioNet data")
    parser.add_argument("--repo-id", type=str, default="legomaheggo/physionet-sepsis-2019",
                        help="HuggingFace dataset repository ID")
    parser.add_argument("--max-patients", type=int, default=None, help="Max patients to upload (for testing)")
    args = parser.parse_args()

    # Import here to avoid issues if not installed
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    data_dir = Path(args.data_dir)

    # Collect all PSV files
    psv_files = []
    for subdir in ["training_setA", "training_setB"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            psv_files.extend(sorted(subdir_path.glob("*.psv")))

    if not psv_files:
        print(f"No PSV files found in {data_dir}")
        return

    if args.max_patients:
        psv_files = psv_files[:args.max_patients]

    print(f"Found {len(psv_files)} patient files")

    # Load all patients
    records = []
    for psv_file in tqdm(psv_files, desc="Loading patients"):
        try:
            record = load_psv_file(psv_file)
            records.append(record)
        except Exception as e:
            print(f"Warning: Failed to load {psv_file}: {e}")

    print(f"Loaded {len(records)} patients")

    # Create dataset
    dataset = Dataset.from_list(records)

    # Split into train (for reference, all data is actually test/validation for our app)
    dataset_dict = DatasetDict({
        "train": dataset
    })

    print(f"Dataset created with {len(dataset)} patients")
    print(f"Features: {dataset.features}")

    # Push to HuggingFace
    print(f"Uploading to {args.repo_id}...")
    dataset_dict.push_to_hub(
        args.repo_id,
        token=args.token,
        private=False,  # Make it public for the app to access
    )

    print(f"Successfully uploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
