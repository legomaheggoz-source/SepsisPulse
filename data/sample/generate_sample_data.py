"""
Generate synthetic sample patient data for SepsisPulse demos.
Creates 20 patients (10 sepsis, 10 non-sepsis) with realistic vital signs and lab values.
"""

import numpy as np
import pandas as pd
import json
import os
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# PhysioNet Challenge 2019 columns (41 total)
COLUMNS = [
    # Vital signs (8)
    'HR',        # Heart rate (beats per minute)
    'O2Sat',     # Pulse oximetry (%)
    'Temp',      # Temperature (Deg C)
    'SBP',       # Systolic BP (mm Hg)
    'MAP',       # Mean arterial pressure (mm Hg)
    'DBP',       # Diastolic BP (mm Hg)
    'Resp',      # Respiration rate (breaths per minute)
    'EtCO2',     # End tidal carbon dioxide (mm Hg)

    # Laboratory values (26)
    'BaseExcess',     # Measure of excess bicarbonate (mmol/L)
    'HCO3',           # Bicarbonate (mmol/L)
    'FiO2',           # Fraction of inspired oxygen (%)
    'pH',             # N/A
    'PaCO2',          # Partial pressure of carbon dioxide from arterial blood (mm Hg)
    'SaO2',           # Oxygen saturation from arterial blood (%)
    'AST',            # Aspartate transaminase (IU/L)
    'BUN',            # Blood urea nitrogen (mg/dL)
    'Alkalinephos',   # Alkaline phosphatase (IU/L)
    'Calcium',        # (mg/dL)
    'Chloride',       # (mmol/L)
    'Creatinine',     # (mg/dL)
    'Bilirubin_direct',  # Bilirubin direct (mg/dL)
    'Glucose',        # Serum glucose (mg/dL)
    'Lactate',        # Lactic acid (mg/dL)
    'Magnesium',      # (mmol/dL)
    'Phosphate',      # (mg/dL)
    'Potassium',      # (mmol/L)
    'Bilirubin_total',   # Total bilirubin (mg/dL)
    'TroponinI',      # Troponin I (ng/mL)
    'Hct',            # Hematocrit (%)
    'Hgb',            # Hemoglobin (g/dL)
    'PTT',            # Partial thromboplastin time (seconds)
    'WBC',            # Leukocyte count (count*10^3/uL)
    'Fibrinogen',     # (mg/dL)
    'Platelets',      # (count*10^3/uL)

    # Demographics (6)
    'Age',            # Years (100 for patients 90 or above)
    'Gender',         # Female (0) or Male (1)
    'Unit1',          # Administrative identifier for ICU unit (MICU)
    'Unit2',          # Administrative identifier for ICU unit (SICU)
    'HospAdmTime',    # Hours between hospital admit and ICU admit
    'ICULOS',         # ICU length-of-stay (hours since ICU admit)

    # Label (1)
    'SepsisLabel'     # For sepsis patients, 1 at and after sepsis onset
]

# Realistic ranges for vital signs and labs
VITAL_RANGES = {
    'HR': {'normal': (60, 100), 'sepsis': (90, 140), 'mean': 80, 'std': 10},
    'O2Sat': {'normal': (95, 100), 'sepsis': (88, 96), 'mean': 97, 'std': 2},
    'Temp': {'normal': (36.0, 37.5), 'sepsis': (38.0, 40.0), 'mean': 37.0, 'std': 0.5},
    'SBP': {'normal': (100, 140), 'sepsis': (70, 100), 'mean': 120, 'std': 15},
    'MAP': {'normal': (70, 100), 'sepsis': (50, 70), 'mean': 85, 'std': 10},
    'DBP': {'normal': (60, 90), 'sepsis': (40, 60), 'mean': 75, 'std': 10},
    'Resp': {'normal': (12, 20), 'sepsis': (22, 35), 'mean': 16, 'std': 3},
    'EtCO2': {'normal': (35, 45), 'sepsis': (25, 35), 'mean': 40, 'std': 5},
}

LAB_RANGES = {
    'BaseExcess': {'normal': (-2, 2), 'sepsis': (-10, -2), 'mean': 0, 'std': 2},
    'HCO3': {'normal': (22, 26), 'sepsis': (15, 22), 'mean': 24, 'std': 3},
    'FiO2': {'normal': (0.21, 0.4), 'sepsis': (0.4, 1.0), 'mean': 0.3, 'std': 0.1},
    'pH': {'normal': (7.35, 7.45), 'sepsis': (7.2, 7.35), 'mean': 7.4, 'std': 0.05},
    'PaCO2': {'normal': (35, 45), 'sepsis': (25, 35), 'mean': 40, 'std': 5},
    'SaO2': {'normal': (95, 100), 'sepsis': (85, 95), 'mean': 97, 'std': 2},
    'AST': {'normal': (10, 40), 'sepsis': (40, 200), 'mean': 25, 'std': 10},
    'BUN': {'normal': (7, 20), 'sepsis': (25, 80), 'mean': 15, 'std': 5},
    'Alkalinephos': {'normal': (44, 147), 'sepsis': (100, 300), 'mean': 80, 'std': 30},
    'Calcium': {'normal': (8.5, 10.5), 'sepsis': (7.0, 8.5), 'mean': 9.5, 'std': 0.5},
    'Chloride': {'normal': (96, 106), 'sepsis': (90, 96), 'mean': 101, 'std': 3},
    'Creatinine': {'normal': (0.7, 1.3), 'sepsis': (1.5, 5.0), 'mean': 1.0, 'std': 0.2},
    'Bilirubin_direct': {'normal': (0.0, 0.3), 'sepsis': (0.5, 5.0), 'mean': 0.1, 'std': 0.1},
    'Glucose': {'normal': (70, 110), 'sepsis': (140, 300), 'mean': 100, 'std': 20},
    'Lactate': {'normal': (0.5, 2.0), 'sepsis': (2.5, 10.0), 'mean': 1.2, 'std': 0.5},
    'Magnesium': {'normal': (1.7, 2.2), 'sepsis': (1.2, 1.7), 'mean': 2.0, 'std': 0.2},
    'Phosphate': {'normal': (2.5, 4.5), 'sepsis': (1.5, 2.5), 'mean': 3.5, 'std': 0.5},
    'Potassium': {'normal': (3.5, 5.0), 'sepsis': (5.0, 6.5), 'mean': 4.2, 'std': 0.4},
    'Bilirubin_total': {'normal': (0.1, 1.2), 'sepsis': (1.5, 10.0), 'mean': 0.7, 'std': 0.3},
    'TroponinI': {'normal': (0.0, 0.04), 'sepsis': (0.1, 2.0), 'mean': 0.02, 'std': 0.01},
    'Hct': {'normal': (36, 50), 'sepsis': (25, 35), 'mean': 42, 'std': 5},
    'Hgb': {'normal': (12, 17), 'sepsis': (8, 11), 'mean': 14, 'std': 2},
    'PTT': {'normal': (25, 35), 'sepsis': (40, 80), 'mean': 30, 'std': 5},
    'WBC': {'normal': (4, 11), 'sepsis': (15, 35), 'mean': 8, 'std': 2},
    'Fibrinogen': {'normal': (200, 400), 'sepsis': (100, 200), 'mean': 300, 'std': 50},
    'Platelets': {'normal': (150, 400), 'sepsis': (50, 150), 'mean': 250, 'std': 50},
}


def generate_vital_sign(name, hours, is_sepsis, onset_hour=None):
    """Generate a time series for a vital sign."""
    ranges = VITAL_RANGES[name]
    values = np.zeros(hours)

    for h in range(hours):
        if is_sepsis and onset_hour and h >= onset_hour:
            # Gradual deterioration towards sepsis range
            progress = min(1.0, (h - onset_hour) / 12)  # Full effect over 12 hours
            low = ranges['normal'][0] + (ranges['sepsis'][0] - ranges['normal'][0]) * progress
            high = ranges['normal'][1] + (ranges['sepsis'][1] - ranges['normal'][1]) * progress
        else:
            low, high = ranges['normal']

        # Add some autocorrelation
        if h == 0:
            values[h] = np.random.uniform(low, high)
        else:
            noise = np.random.normal(0, ranges['std'] * 0.3)
            values[h] = np.clip(values[h-1] + noise, low, high)

    # Vital signs have low missing rate (~10%)
    mask = np.random.random(hours) < 0.1
    values[mask] = np.nan

    return values


def generate_lab_value(name, hours, is_sepsis, onset_hour=None):
    """Generate a time series for a lab value with high missing rate."""
    ranges = LAB_RANGES[name]
    values = np.full(hours, np.nan)

    # Labs measured infrequently: 2-6 times during stay
    num_measurements = np.random.randint(2, 7)
    measurement_hours = sorted(np.random.choice(hours, size=min(num_measurements, hours), replace=False))

    for h in measurement_hours:
        if is_sepsis and onset_hour and h >= onset_hour:
            # Values shift towards sepsis range
            progress = min(1.0, (h - onset_hour) / 12)
            low = ranges['normal'][0] + (ranges['sepsis'][0] - ranges['normal'][0]) * progress
            high = ranges['normal'][1] + (ranges['sepsis'][1] - ranges['normal'][1]) * progress
        else:
            low, high = ranges['normal']

        values[h] = np.random.uniform(low, high)

    return values


def generate_patient(patient_id, is_sepsis):
    """Generate a single patient's data."""
    # Random stay length: 24-72 hours
    hours = np.random.randint(24, 73)

    # For sepsis patients, onset between hour 24-48
    onset_hour = None
    if is_sepsis:
        onset_hour = np.random.randint(24, min(49, hours - 6))  # Ensure at least 6 hours after onset

    # Demographics (constant throughout stay)
    age = np.random.randint(18, 90)
    gender = np.random.randint(0, 2)
    unit1 = np.random.randint(0, 2)
    unit2 = 1 - unit1  # Either MICU or SICU
    hosp_adm_time = np.random.uniform(-72, 0)  # Hours before ICU admit

    # Create DataFrame
    data = {col: np.full(hours, np.nan) for col in COLUMNS}

    # Generate vital signs
    for vital in VITAL_RANGES.keys():
        data[vital] = generate_vital_sign(vital, hours, is_sepsis, onset_hour)

    # Generate lab values
    for lab in LAB_RANGES.keys():
        data[lab] = generate_lab_value(lab, hours, is_sepsis, onset_hour)

    # Set demographics (constant)
    data['Age'] = np.full(hours, age)
    data['Gender'] = np.full(hours, gender)
    data['Unit1'] = np.full(hours, unit1)
    data['Unit2'] = np.full(hours, unit2)
    data['HospAdmTime'] = np.full(hours, hosp_adm_time)
    data['ICULOS'] = np.arange(1, hours + 1)

    # Set SepsisLabel
    if is_sepsis:
        data['SepsisLabel'] = np.where(np.arange(hours) >= onset_hour, 1, 0)
    else:
        data['SepsisLabel'] = np.zeros(hours)

    df = pd.DataFrame(data)
    return df, hours, onset_hour


def main():
    output_dir = r'C:\SepsisPulse\data\sample\patients'
    os.makedirs(output_dir, exist_ok=True)

    total_hours = 0
    patient_info = []

    # Generate 10 sepsis patients (p00001-p00010) and 10 non-sepsis (p00011-p00020)
    for i in range(1, 21):
        is_sepsis = i <= 10
        patient_id = f'p{i:05d}'

        df, hours, onset_hour = generate_patient(patient_id, is_sepsis)
        total_hours += hours

        # Save as PSV (pipe-separated values)
        filepath = os.path.join(output_dir, f'{patient_id}.psv')
        df.to_csv(filepath, sep='|', index=False, na_rep='NaN')

        patient_info.append({
            'patient_id': patient_id,
            'hours': hours,
            'is_sepsis': is_sepsis,
            'onset_hour': onset_hour
        })

        print(f'Generated {patient_id}: {hours} hours, sepsis={is_sepsis}, onset={onset_hour}')

    # Create metadata.json
    metadata = {
        'patient_count': 20,
        'sepsis_count': 10,
        'non_sepsis_count': 10,
        'total_hours': int(total_hours),
        'column_count': len(COLUMNS),
        'column_list': COLUMNS,
        'column_categories': {
            'vital_signs': list(VITAL_RANGES.keys()),
            'laboratory_values': list(LAB_RANGES.keys()),
            'demographics': ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS'],
            'label': ['SepsisLabel']
        },
        'patients': patient_info,
        'data_format': 'PSV (pipe-separated values)',
        'missing_value_representation': 'NaN',
        'generation_seed': 42
    }

    metadata_path = r'C:\SepsisPulse\data\sample\metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'\nSaved metadata to {metadata_path}')

    print(f'\nSummary:')
    print(f'  Total patients: 20')
    print(f'  Sepsis patients: 10')
    print(f'  Non-sepsis patients: 10')
    print(f'  Total hours: {total_hours}')
    print(f'  Columns: {len(COLUMNS)}')


if __name__ == '__main__':
    main()
