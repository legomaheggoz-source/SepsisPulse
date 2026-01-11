# SepsisPulse - User Guide

Welcome to SepsisPulse, a clinical utility auditor for sepsis early warning systems. This guide will help you get started, navigate the dashboard, and interpret the results.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Running the Application](#running-the-application)
4. [Dashboard Navigation](#dashboard-navigation)
5. [Interpreting Results](#interpreting-results)
6. [Configuration Options](#configuration-options)
7. [Working with Your Own Data](#working-with-your-own-data)
8. [FAQ](#faq)

---

## Getting Started

### What is SepsisPulse?

SepsisPulse is a tool that compares three different approaches to predicting sepsis:

| Model | Type | Description |
|-------|------|-------------|
| **qSOFA** | Rule-based | Bedside screening tool from Sepsis-3 guidelines |
| **XGBoost-TS** | Machine Learning | Gradient boosted trees with time-series features |
| **TFT-Lite** | Deep Learning | Lightweight Temporal Fusion Transformer |

The key metric is the **Clinical Utility Score**, which rewards early predictions while penalizing false alarms.

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.9 | 3.10+ |
| RAM | 2 GB | 4 GB |
| Disk Space | 500 MB | 1 GB |
| Browser | Any modern browser | Chrome/Firefox |

---

## Installation

### Option 1: Quick Start (pip)

```bash
# Clone the repository
git clone https://github.com/user/SepsisPulse.git
cd SepsisPulse

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/user/SepsisPulse.git
cd SepsisPulse

# Create conda environment
conda create -n sepsispulse python=3.10
conda activate sepsispulse

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check that all modules import correctly
python -c "from models import QSOFAModel, XGBoostTSModel, TFTLiteModel; print('Success!')"
```

---

## Running the Application

### Start the Dashboard

```bash
# From the SepsisPulse directory
streamlit run app.py
```

The application will start and open in your default browser at `http://localhost:8501`.

### Command Line Options

```bash
# Run on a different port
streamlit run app.py --server.port 8080

# Run without opening browser
streamlit run app.py --server.headless true

# Enable debug mode
streamlit run app.py --logger.level debug
```

### Stopping the Application

Press `Ctrl+C` in the terminal to stop the Streamlit server.

---

## Dashboard Navigation

The SepsisPulse dashboard has five main sections, accessible from the sidebar.

### 1. Dashboard (Main View)

The main dashboard provides an overview of model performance.

**Key Metrics Row:**
- **Total Patients**: Number of patients in the dataset
- **Sepsis Cases**: Count and percentage of sepsis patients
- **Best Lead Time**: Highest mean lead time across models
- **Best Utility Score**: Highest clinical utility score

**Model Comparison Cards:**
Three cards showing each model's key metrics:
- AUC-ROC score
- Clinical Utility Score
- Mean Lead Time

The best-performing model is highlighted with a border glow.

**Performance Charts:**
- **ROC Curves**: Compare discriminative ability across models
- **Lead Time Distribution**: Histogram of prediction timing
- **Utility vs Threshold**: How utility changes with decision threshold

---

### 2. Model Comparison

Detailed side-by-side comparison of all three models.

**Metrics Table:**
| Metric | qSOFA | XGBoost-TS | TFT-Lite |
|--------|-------|------------|----------|
| AUC-ROC | 0.72 | 0.85 | 0.87 |
| Utility | 0.31 | 0.39 | 0.42 |
| Lead Time | 4.1h | 5.8h | 6.2h |
| Sensitivity | 0.65 | 0.79 | 0.82 |
| Specificity | 0.78 | 0.88 | 0.89 |

**Model Details Tabs:**
- **qSOFA**: Criteria definitions, scoring rules
- **XGBoost-TS**: Architecture, feature engineering details
- **TFT-Lite**: Neural network architecture, parameter counts

---

### 3. Patient Explorer

Explore individual patient data and predictions.

**Patient Selection:**
- Dropdown to select patient ID
- Patient demographics (age, gender, ICU stay duration)
- Sepsis status indicator

**Vital Signs Timeline:**
- Heart Rate, Blood Pressure, Temperature, Respiratory Rate
- Normal ranges shown as shaded regions
- Interactive zoom and pan

**Model Predictions:**
- Risk probability from each model
- Risk level indicator (Low/Medium/High)
- Color-coded by severity

---

### 4. Configuration

Adjust prediction thresholds and display settings.

**Prediction Thresholds:**
- qSOFA threshold (1-3)
- XGBoost-TS threshold (0.0-1.0)
- TFT-Lite threshold (0.0-1.0)

**Lead Time Settings:**
- Optimal lead time target (hours)
- Maximum useful lead time (hours)

**Display Settings:**
- Show confidence intervals
- Include all patients
- Color theme selection

**Data Settings:**
- Data source selection (Sample/Full/Custom)
- Upload custom PSV files

---

### 5. Documentation

In-app reference documentation.

**Tabs:**
- **Overview**: System introduction, key concepts
- **Clinical Utility Score**: Detailed explanation of the metric
- **Models**: Architecture descriptions
- **Data Format**: PhysioNet 2019 column definitions

---

## Interpreting Results

### Clinical Utility Score

The Clinical Utility Score is the primary metric for comparing models. It reflects clinical priorities:

| Outcome | Utility Value | Meaning |
|---------|---------------|---------|
| Optimal Prediction | +1.0 | Predicted 6-12 hours before sepsis |
| Acceptable Prediction | +0.5 | Predicted 0-6 hours before sepsis |
| Too Early | -0.05 | Predicted >12 hours before sepsis |
| Missed | -2.0 | Failed to predict before sepsis onset |
| False Positive | -0.05 | Predicted sepsis for non-sepsis patient |

**Interpretation:**
- Score of 0.0 = No better than never predicting sepsis
- Score of 0.4+ = Good performance
- Score of 0.5+ = Excellent performance

### Lead Time

Lead time measures how early the model predicts sepsis before onset.

**Guidelines:**
- **< 3 hours**: Too late for optimal intervention
- **3-6 hours**: Acceptable, allows treatment initiation
- **6-12 hours**: Optimal, maximizes intervention window
- **> 12 hours**: May increase false positive risk

### AUC-ROC

Area Under the ROC Curve measures discriminative ability.

**Guidelines:**
- **0.5**: Random chance (no discrimination)
- **0.7-0.8**: Acceptable discrimination
- **0.8-0.9**: Good discrimination
- **> 0.9**: Excellent discrimination

### Sensitivity vs Specificity Trade-off

| Priority | Adjust Threshold | Result |
|----------|------------------|--------|
| Catch more sepsis cases | Lower threshold | Higher sensitivity, more false alarms |
| Reduce false alarms | Raise threshold | Higher specificity, may miss cases |
| Balance | Optimize for utility | Maximize clinical utility score |

---

## Configuration Options

### Environment Variables

You can configure SepsisPulse using environment variables:

```bash
# Set data directory
export SEPSISPULSE_DATA_DIR=/path/to/data

# Set prediction threshold
export SEPSISPULSE_PREDICTION_THRESHOLD=0.6

# Set optimal lead time target
export SEPSISPULSE_OPTIMAL_LEAD_TIME=6
```

**Available Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SEPSISPULSE_DATA_DIR` | `./data` | Path to data directory |
| `SEPSISPULSE_MODELS_DIR` | `./models` | Path to model weights |
| `SEPSISPULSE_PREDICTION_THRESHOLD` | `0.5` | Default prediction threshold |
| `SEPSISPULSE_OPTIMAL_LEAD_TIME` | `6` | Target lead time (hours) |
| `SEPSISPULSE_MAX_LEAD_TIME` | `12` | Maximum useful lead time |
| `SEPSISPULSE_QSOFA_THRESHOLD` | `2` | qSOFA score threshold |

### Streamlit Configuration

The `.streamlit/config.toml` file contains Streamlit settings:

```toml
[theme]
primaryColor = "#00d9ff"
backgroundColor = "#0a0e17"
secondaryBackgroundColor = "#002d42"
textColor = "#ffffff"

[server]
maxUploadSize = 50  # MB

[browser]
gatherUsageStats = false
```

---

## Working with Your Own Data

### Data Format Requirements

SepsisPulse expects data in PhysioNet 2019 PSV format:

- **File extension**: `.psv` (pipe-separated values)
- **Delimiter**: `|` (pipe character)
- **Rows**: One row per hour of ICU stay
- **Columns**: 41 columns (see DATA_PREPARATION.md for details)

### Uploading Custom Data

1. Navigate to the **Configuration** page
2. Under **Data Settings**, select "Custom Upload"
3. Click **Browse files** and select your PSV files
4. The dashboard will reload with your data

### Preparing Your Data

If your data is in a different format:

```python
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Rename columns to match PhysioNet format
column_mapping = {
    'heart_rate': 'HR',
    'oxygen_saturation': 'O2Sat',
    'temperature': 'Temp',
    # ... add all 41 columns
}
df = df.rename(columns=column_mapping)

# Save as PSV
df.to_csv("patient_001.psv", sep='|', index=False)
```

---

## FAQ

### General Questions

**Q: What is sepsis?**

A: Sepsis is a life-threatening condition caused by the body's dysregulated response to infection. It can lead to organ failure and death if not treated promptly.

**Q: Why is early prediction important?**

A: Every hour of delayed sepsis treatment increases mortality by approximately 8%. Early prediction allows clinicians to intervene sooner.

**Q: What is alarm fatigue?**

A: Alarm fatigue occurs when clinicians become desensitized to frequent alerts, potentially causing them to miss or ignore critical warnings. SepsisPulse's utility score penalizes false alarms to help address this issue.

---

### Technical Questions

**Q: Why are my models in "demo mode"?**

A: Demo mode activates when pre-trained weights cannot be found. The models will generate random predictions for demonstration purposes. To use real predictions, ensure model weights are in the correct locations:
- XGBoost: `models/xgboost_ts/weights/xgb_sepsis_v1.json`
- TFT-Lite: `models/tft_lite/weights/tft_lite_v1.pt`

**Q: How do I train my own models?**

A: SepsisPulse is designed for inference only. For training:
1. Download the full PhysioNet 2019 dataset
2. Use the preprocessing and feature engineering modules
3. Train using your preferred framework
4. Save weights in the expected format

**Q: Why is the TFT-Lite model smaller than the original?**

A: TFT-Lite was designed to fit within 2GB RAM constraints for free-tier cloud deployment. It uses reduced hidden dimensions (32 vs 256), fewer LSTM layers (1 vs 2), and fewer attention heads (2 vs 4).

**Q: How do I access the full PhysioNet dataset?**

A: See DATA_PREPARATION.md for detailed instructions on downloading the PhysioNet Challenge 2019 dataset.

---

### Troubleshooting

**Problem: Streamlit won't start**

```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall Streamlit
pip uninstall streamlit
pip install streamlit

# Try running with verbose output
streamlit run app.py --logger.level debug
```

**Problem: Out of memory error**

```bash
# Limit number of patients loaded
# In Configuration page, use Sample Subset option

# Or set environment variable
export SEPSISPULSE_MAX_PATIENTS=100
```

**Problem: Charts not displaying**

```bash
# Reinstall Plotly
pip uninstall plotly
pip install plotly

# Clear Streamlit cache
rm -rf ~/.streamlit/cache
```

**Problem: Import errors**

```bash
# Ensure you're in the SepsisPulse directory
cd /path/to/SepsisPulse

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/user/SepsisPulse/issues) page
2. Review the API documentation in API.md
3. Examine the architecture in ARCHITECTURE.md
4. Open a new issue with:
   - Python version
   - Operating system
   - Error message (full traceback)
   - Steps to reproduce
