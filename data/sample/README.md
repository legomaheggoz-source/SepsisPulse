# SepsisPulse Sample Data

This directory contains synthetic patient data for SepsisPulse demos and testing. The data is designed to mimic the format and characteristics of the PhysioNet Computing in Cardiology Challenge 2019 dataset.

## Why Sample Data?

The PhysioNet Challenge 2019 dataset requires registration and approval to access. This synthetic sample data allows developers and users to:

- Test the SepsisPulse pipeline without waiting for PhysioNet access
- Run demos and demonstrations
- Develop and debug features
- Understand the data format and structure

## Data Overview

| Metric | Value |
|--------|-------|
| Total Patients | 20 |
| Sepsis Patients | 10 (p00001 - p00010) |
| Non-Sepsis Patients | 10 (p00011 - p00020) |
| Total Hours | 891 |
| Columns | 41 |
| Data Format | PSV (pipe-separated values) |

## Directory Structure

```
data/sample/
├── patients/           # Individual patient files
│   ├── p00001.psv     # Sepsis patient
│   ├── p00002.psv     # Sepsis patient
│   ├── ...
│   ├── p00010.psv     # Sepsis patient
│   ├── p00011.psv     # Non-sepsis patient
│   ├── ...
│   └── p00020.psv     # Non-sepsis patient
├── metadata.json      # Summary statistics and column information
├── generate_sample_data.py  # Script used to generate this data
└── README.md          # This file
```

## Column Descriptions

### Vital Signs (8 columns)
| Column | Description | Units |
|--------|-------------|-------|
| HR | Heart rate | beats/min |
| O2Sat | Pulse oximetry | % |
| Temp | Temperature | Deg C |
| SBP | Systolic blood pressure | mm Hg |
| MAP | Mean arterial pressure | mm Hg |
| DBP | Diastolic blood pressure | mm Hg |
| Resp | Respiration rate | breaths/min |
| EtCO2 | End tidal carbon dioxide | mm Hg |

### Laboratory Values (26 columns)
| Column | Description | Units |
|--------|-------------|-------|
| BaseExcess | Excess bicarbonate | mmol/L |
| HCO3 | Bicarbonate | mmol/L |
| FiO2 | Fraction of inspired oxygen | % |
| pH | Blood pH | - |
| PaCO2 | Partial pressure of CO2 | mm Hg |
| SaO2 | Arterial oxygen saturation | % |
| AST | Aspartate transaminase | IU/L |
| BUN | Blood urea nitrogen | mg/dL |
| Alkalinephos | Alkaline phosphatase | IU/L |
| Calcium | Calcium | mg/dL |
| Chloride | Chloride | mmol/L |
| Creatinine | Creatinine | mg/dL |
| Bilirubin_direct | Direct bilirubin | mg/dL |
| Glucose | Serum glucose | mg/dL |
| Lactate | Lactic acid | mg/dL |
| Magnesium | Magnesium | mmol/dL |
| Phosphate | Phosphate | mg/dL |
| Potassium | Potassium | mmol/L |
| Bilirubin_total | Total bilirubin | mg/dL |
| TroponinI | Troponin I | ng/mL |
| Hct | Hematocrit | % |
| Hgb | Hemoglobin | g/dL |
| PTT | Partial thromboplastin time | seconds |
| WBC | White blood cell count | 10^3/uL |
| Fibrinogen | Fibrinogen | mg/dL |
| Platelets | Platelet count | 10^3/uL |

### Demographics (6 columns)
| Column | Description |
|--------|-------------|
| Age | Patient age in years |
| Gender | 0 = Female, 1 = Male |
| Unit1 | MICU indicator (0 or 1) |
| Unit2 | SICU indicator (0 or 1) |
| HospAdmTime | Hours between hospital and ICU admission |
| ICULOS | ICU length of stay (hours since ICU admit) |

### Label (1 column)
| Column | Description |
|--------|-------------|
| SepsisLabel | 0 = No sepsis, 1 = Sepsis (at or after onset) |

## Data Characteristics

### Realistic Properties
- **Vital signs**: Measured hourly with ~10% missing rate
- **Lab values**: High missing rate (>80%) as labs are measured infrequently
- **Sepsis progression**: Gradual deterioration of vitals/labs starting at onset
- **ICU stay length**: 24-72 hours per patient
- **Sepsis onset**: Between hours 24-48 for sepsis patients

### Value Ranges

**Normal (Non-Sepsis) Ranges:**
- Heart Rate: 60-100 bpm
- O2 Saturation: 95-100%
- Temperature: 36.0-37.5 C
- Systolic BP: 100-140 mm Hg
- Lactate: 0.5-2.0 mg/dL
- WBC: 4-11 x10^3/uL

**Sepsis Progression Ranges:**
- Heart Rate: 90-140 bpm (elevated)
- O2 Saturation: 88-96% (decreased)
- Temperature: 38.0-40.0 C (fever)
- Systolic BP: 70-100 mm Hg (hypotension)
- Lactate: 2.5-10.0 mg/dL (elevated)
- WBC: 15-35 x10^3/uL (elevated)

## Usage

### Loading a Single Patient
```python
import pandas as pd

patient = pd.read_csv('patients/p00001.psv', sep='|')
print(patient.shape)  # (62, 41)
print(patient['SepsisLabel'].sum())  # Hours with sepsis
```

### Loading All Patients
```python
import pandas as pd
import glob

patients = []
for f in glob.glob('patients/*.psv'):
    df = pd.read_csv(f, sep='|')
    df['patient_id'] = f.split('/')[-1].replace('.psv', '')
    patients.append(df)

all_data = pd.concat(patients, ignore_index=True)
print(all_data.shape)  # (891, 42)
```

### Loading Metadata
```python
import json

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Patients: {metadata['patient_count']}")
print(f"Sepsis cases: {metadata['sepsis_count']}")
print(f"Total hours: {metadata['total_hours']}")
```

## Regenerating Data

To regenerate the sample data with a different seed or configuration:

```bash
python generate_sample_data.py
```

The script uses a fixed random seed (42) for reproducibility.

## Important Notes

1. **This is synthetic data** - Not real patient data. Patterns and correlations are simplified approximations of real clinical data.

2. **For development only** - This data should be used for demos, testing, and development. For actual model training and validation, use the real PhysioNet dataset.

3. **Missing values** - Represented as `NaN` in the PSV files. Handle appropriately during data loading.

4. **Class balance** - The dataset is perfectly balanced (50% sepsis). Real data may have significant class imbalance.
