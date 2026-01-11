# SepsisPulse - Data Preparation Guide

This document describes the PhysioNet 2019 Challenge data format and provides instructions for downloading the full dataset or preparing custom data.

---

## Table of Contents

1. [PhysioNet 2019 Format Description](#physionet-2019-format-description)
2. [Column Descriptions](#column-descriptions)
3. [Downloading the Full Dataset](#downloading-the-full-dataset)
4. [Adding Custom Data](#adding-custom-data)
5. [Data Quality Considerations](#data-quality-considerations)

---

## PhysioNet 2019 Format Description

### Overview

The PhysioNet/Computing in Cardiology Challenge 2019 focused on early prediction of sepsis from clinical data. The dataset contains de-identified ICU patient records with hourly measurements.

### File Format

| Property | Value |
|----------|-------|
| **Extension** | `.psv` (pipe-separated values) |
| **Delimiter** | `\|` (pipe character) |
| **Encoding** | UTF-8 |
| **Missing Values** | Empty string or `NaN` |
| **Structure** | One file per patient |
| **Naming** | `pXXXXX.psv` (e.g., p00001.psv) |

### Data Structure

- **Rows**: One row per hour of ICU stay
- **Columns**: 41 features (8 vitals + 26 labs + 6 demographics + 1 label)
- **Time Index**: Implicit (row 0 = hour 0, row 1 = hour 1, etc.)

### Example File

```
HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|...|SepsisLabel
82|97|36.8|118|78|58|16|NaN|NaN|...|0
85|96|36.9|115|76|56|18|NaN|NaN|...|0
88|95|37.2|112|74|54|20|NaN|0.5|...|0
92|94|37.8|108|72|52|22|NaN|NaN|...|0
96|93|38.2|105|70|50|24|NaN|-2.0|...|1
```

---

## Column Descriptions

### Vital Signs (8 columns)

Measured hourly (continuous monitoring).

| Column | Name | Units | Normal Range | Description |
|--------|------|-------|--------------|-------------|
| `HR` | Heart Rate | beats/min | 60-100 | Cardiac rhythm |
| `O2Sat` | Oxygen Saturation | % | 95-100 | Pulse oximetry (SpO2) |
| `Temp` | Temperature | Celsius | 36.1-37.2 | Core body temperature |
| `SBP` | Systolic Blood Pressure | mmHg | 90-140 | Peak arterial pressure |
| `MAP` | Mean Arterial Pressure | mmHg | 70-100 | Average arterial pressure |
| `DBP` | Diastolic Blood Pressure | mmHg | 60-90 | Minimum arterial pressure |
| `Resp` | Respiratory Rate | breaths/min | 12-20 | Breathing frequency |
| `EtCO2` | End-Tidal CO2 | mmHg | 35-45 | Exhaled carbon dioxide |

**Missing Rates**: ~10-20% (equipment disconnection, patient movement)

---

### Laboratory Values (26 columns)

Measured when clinically indicated (intermittent).

| Column | Name | Units | Normal Range | Description |
|--------|------|-------|--------------|-------------|
| `BaseExcess` | Base Excess | mmol/L | -2 to +2 | Acid-base balance |
| `HCO3` | Bicarbonate | mmol/L | 22-26 | Blood buffer |
| `FiO2` | Fraction Inspired O2 | % | 21 (room air) | Supplemental oxygen |
| `pH` | Blood pH | - | 7.35-7.45 | Acidity level |
| `PaCO2` | Partial Pressure CO2 | mmHg | 35-45 | Arterial carbon dioxide |
| `SaO2` | Arterial O2 Saturation | % | 95-100 | Oxygen in arterial blood |
| `AST` | Aspartate Transaminase | IU/L | 10-40 | Liver enzyme |
| `BUN` | Blood Urea Nitrogen | mg/dL | 7-20 | Kidney function |
| `Alkalinephos` | Alkaline Phosphatase | IU/L | 44-147 | Liver/bone enzyme |
| `Calcium` | Calcium | mg/dL | 8.5-10.5 | Electrolyte |
| `Chloride` | Chloride | mmol/L | 96-106 | Electrolyte |
| `Creatinine` | Creatinine | mg/dL | 0.7-1.3 | Kidney function |
| `Bilirubin_direct` | Direct Bilirubin | mg/dL | 0-0.3 | Liver function |
| `Glucose` | Glucose | mg/dL | 70-100 | Blood sugar |
| `Lactate` | Lactate | mmol/L | 0.5-2.0 | Tissue perfusion |
| `Magnesium` | Magnesium | mmol/dL | 1.5-2.5 | Electrolyte |
| `Phosphate` | Phosphate | mg/dL | 2.5-4.5 | Electrolyte |
| `Potassium` | Potassium | mmol/L | 3.5-5.0 | Electrolyte |
| `Bilirubin_total` | Total Bilirubin | mg/dL | 0.1-1.2 | Liver function |
| `TroponinI` | Troponin I | ng/mL | <0.04 | Cardiac damage marker |
| `Hct` | Hematocrit | % | 36-44 | Red blood cell volume |
| `Hgb` | Hemoglobin | g/dL | 12-16 | Oxygen-carrying capacity |
| `PTT` | Partial Thromboplastin Time | seconds | 25-35 | Clotting function |
| `WBC` | White Blood Cell Count | 10^3/uL | 4-11 | Immune response |
| `Fibrinogen` | Fibrinogen | mg/dL | 200-400 | Clotting protein |
| `Platelets` | Platelet Count | 10^3/uL | 150-400 | Clotting cells |

**Missing Rates**: 80-98% (labs only drawn when clinically indicated)

---

### Demographics (6 columns)

Static patient characteristics.

| Column | Name | Values | Description |
|--------|------|--------|-------------|
| `Age` | Age | 18-100+ | Patient age (100 = 90+) |
| `Gender` | Gender | 0, 1 | 0 = Female, 1 = Male |
| `Unit1` | ICU Unit 1 | 0, 1 | MICU indicator |
| `Unit2` | ICU Unit 2 | 0, 1 | SICU indicator |
| `HospAdmTime` | Hospital Admit Time | hours | Time between hospital and ICU admission |
| `ICULOS` | ICU Length of Stay | hours | Hours since ICU admission (row index) |

**Notes**:
- `Unit1` and `Unit2` are mutually exclusive (patient is in MICU xor SICU)
- `ICULOS` increments by 1 each row (hour 0, 1, 2, ...)
- `HospAdmTime` can be negative (ICU before hospital admission, e.g., emergency)

---

### Label (1 column)

| Column | Name | Values | Description |
|--------|------|--------|-------------|
| `SepsisLabel` | Sepsis Label | 0, 1 | 0 = No sepsis, 1 = Sepsis (at or after onset) |

**Important Notes**:
- Label becomes 1 at sepsis onset and remains 1 for all subsequent hours
- Sepsis onset is defined as the first hour when the Sepsis-3 criteria are met
- For non-sepsis patients, label is 0 for all hours

---

## Downloading the Full Dataset

### Step 1: Create PhysioNet Account

1. Go to https://physionet.org/register/
2. Create an account with a valid email
3. Verify your email address

### Step 2: Accept Data Use Agreement

1. Navigate to the dataset page:
   https://physionet.org/content/challenge-2019/1.0.0/
2. Read the data use agreement
3. Click "Request Access" and accept the terms

### Step 3: Download Data

**Option A: Web Download**

1. Once approved, click "Download" on the dataset page
2. Select the files you need:
   - `training_setA.zip` (Set A: 20,336 patients)
   - `training_setB.zip` (Set B: 20,000 patients)

**Option B: Command Line (wget)**

```bash
# Authenticate with PhysioNet credentials
export PHYSIONET_USER=your_username
export PHYSIONET_PASS=your_password

# Download Set A
wget -r -N -c -np --user $PHYSIONET_USER --password $PHYSIONET_PASS \
  https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/

# Download Set B
wget -r -N -c -np --user $PHYSIONET_USER --password $PHYSIONET_PASS \
  https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/
```

### Step 4: Extract and Organize

```bash
# Extract downloaded files
unzip training_setA.zip -d data/
unzip training_setB.zip -d data/

# Expected structure:
# data/
# +-- training_setA/
# |   +-- p000001.psv
# |   +-- p000002.psv
# |   +-- ...
# +-- training_setB/
#     +-- p100001.psv
#     +-- ...
```

### Step 5: Verify Data

```python
from src.data.loader import load_dataset

# Load first 100 patients to verify
dataset = load_dataset("data/training_setA", max_patients=100)
print(f"Loaded {len(dataset)} patients")
print(f"Columns: {len(dataset[list(dataset.keys())[0]].columns)}")
# Should print: Loaded 100 patients
# Should print: Columns: 41
```

---

## Adding Custom Data

### Requirements

Your data must be converted to PhysioNet 2019 format:

1. One PSV file per patient
2. Pipe (`|`) delimiter
3. 41 columns in the correct order
4. One row per hour of observation

### Conversion Script

```python
import pandas as pd
import os
from pathlib import Path

def convert_to_physionet_format(
    input_df: pd.DataFrame,
    patient_id: str,
    output_dir: str,
    column_mapping: dict
) -> None:
    """
    Convert a DataFrame to PhysioNet 2019 PSV format.

    Parameters
    ----------
    input_df : pd.DataFrame
        Input patient data
    patient_id : str
        Patient identifier (e.g., 'p00001')
    output_dir : str
        Output directory for PSV files
    column_mapping : dict
        Mapping from your column names to PhysioNet column names
    """
    # Expected columns in order
    PHYSIONET_COLUMNS = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
        'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
        'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
        'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
        'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
        'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
        'HospAdmTime', 'ICULOS', 'SepsisLabel'
    ]

    # Rename columns
    df = input_df.rename(columns=column_mapping)

    # Add missing columns with NaN
    for col in PHYSIONET_COLUMNS:
        if col not in df.columns:
            df[col] = float('nan')

    # Reorder columns
    df = df[PHYSIONET_COLUMNS]

    # Ensure ICULOS is set correctly
    df['ICULOS'] = range(len(df))

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save as PSV
    output_path = os.path.join(output_dir, f"{patient_id}.psv")
    df.to_csv(output_path, sep='|', index=False)
    print(f"Saved {output_path}")


# Example usage
column_mapping = {
    'heart_rate': 'HR',
    'spo2': 'O2Sat',
    'temperature': 'Temp',
    'systolic_bp': 'SBP',
    'mean_arterial_pressure': 'MAP',
    'diastolic_bp': 'DBP',
    'respiratory_rate': 'Resp',
    'age': 'Age',
    'sex': 'Gender',  # Make sure 0=Female, 1=Male
    'sepsis': 'SepsisLabel',
}

# Load your data
my_data = pd.read_csv("my_patient_data.csv")

# Convert
convert_to_physionet_format(
    input_df=my_data,
    patient_id="p00001",
    output_dir="data/custom",
    column_mapping=column_mapping
)
```

### Validation

After conversion, validate your data:

```python
from src.data.loader import load_patient
from src.data.preprocessor import get_missing_rate

# Load converted file
df = load_patient("data/custom/p00001.psv")

# Check structure
print(f"Shape: {df.shape}")  # Should be (N, 41)
print(f"Columns: {len(df.columns)}")  # Should be 41

# Check missing rates
missing = get_missing_rate(df)
print("Missing rates:")
print(missing.describe())

# Check vital signs are in reasonable ranges
print("\nVital sign ranges:")
print(f"HR: {df['HR'].min():.1f} - {df['HR'].max():.1f} (expected: 30-200)")
print(f"Temp: {df['Temp'].min():.1f} - {df['Temp'].max():.1f} (expected: 35-42)")
print(f"SBP: {df['SBP'].min():.1f} - {df['SBP'].max():.1f} (expected: 50-250)")
```

---

## Data Quality Considerations

### Missing Values

PhysioNet data has extensive missing values by design:

| Feature Type | Expected Missing Rate |
|--------------|----------------------|
| Vital Signs | 10-20% |
| Laboratory Values | 80-98% |
| Demographics | 0% |
| SepsisLabel | 0% |

**Why so many missing labs?**

Laboratory values are only measured when clinically indicated. A patient who appears stable may not have labs drawn for hours or days. This is **not** a data quality issue - it reflects real clinical practice.

### Handling Missing Values

SepsisPulse uses a three-step imputation strategy:

1. **Forward Fill**: Carry last observation forward (clinically appropriate for labs)
2. **Backward Fill**: Fill initial missing values with first available measurement
3. **Mean Imputation**: Use population means for entirely missing columns

```python
from src.data.preprocessor import handle_missing_values

# Before imputation
print(f"Missing before: {df.isna().sum().sum()}")

# After imputation
df_clean = handle_missing_values(df)
print(f"Missing after: {df_clean.isna().sum().sum()}")  # Should be 0
```

### Data Leakage Warning

**Do not use future information to predict the past.**

Common leakage issues:
- Using labels from future hours in feature engineering
- Normalizing with statistics computed on full dataset (including test set)
- Using `bfill()` with labels (would reveal sepsis before onset)

The preprocessing module is designed to avoid leakage:
- Forward fill is causal (uses past values only)
- Backward fill is only applied to initial missing values
- Normalization uses pre-computed population statistics

### Sepsis Label Definition

The `SepsisLabel` column indicates sepsis status:

- `0` = No sepsis at this hour
- `1` = Sepsis onset or post-onset

**Critical: The label becomes 1 at onset and stays 1.**

This means:
- For prediction, you should predict `1` **before** the label becomes `1`
- The label indicates **current status**, not future risk
- A "lead time" of 6 hours means predicting at hour `t` when label becomes `1` at hour `t+6`

### Class Imbalance

The PhysioNet dataset has significant class imbalance:

| Metric | Value |
|--------|-------|
| Total Patients | 40,336 |
| Sepsis Patients | ~2,932 (7.3%) |
| Total Hours | ~1.5M |
| Sepsis Hours | ~40K (2.7%) |

**Implications**:
- High accuracy can be achieved by always predicting "no sepsis"
- AUC-ROC is a better metric than accuracy
- Clinical utility score accounts for imbalance through its weighting scheme

---

## Quick Reference

### Column Order

```python
PHYSIONET_COLUMNS = [
    # Vital Signs (8)
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
    # Laboratory Values (26)
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
    'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
    'Fibrinogen', 'Platelets',
    # Demographics (6)
    'Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS',
    # Label (1)
    'SepsisLabel'
]
```

### Loading Data

```python
from src.data.loader import load_patient, load_dataset, get_sample_subset

# Single patient
df = load_patient("data/sample/patients/p00001.psv")

# Multiple patients
dataset = load_dataset("data/training_setA", max_patients=1000)

# Sample data (bundled)
sample = get_sample_subset()
```

### Preprocessing

```python
from src.data.preprocessor import preprocess_patient

# Full pipeline (imputation + normalization)
processed = preprocess_patient(df, normalize=True)

# Just imputation (keep raw values)
processed = preprocess_patient(df, normalize=False)
```
