---
title: SepsisPulse
emoji: "📊"
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
---

# SepsisPulse

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/legomaheggo/SepsisPulse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/legomaheggo/SepsisPulse)

**Clinical Utility & Lead-Time Auditor for Sepsis Early Warning**

SepsisPulse is an interactive auditing platform that compares three mathematical approaches to early sepsis prediction: the qSOFA rule-based heuristic, an XGBoost time-series model, and a lightweight Temporal Fusion Transformer. Built for clinicians and ML researchers, it evaluates models using the PhysioNet Challenge 2019 clinical utility score—rewarding timely detections while penalizing false alarms and late predictions. Deployed on HuggingFace Spaces with a modern Aurora-themed interface.

## Screenshot

[Demo Dashboard Screenshot - Shows the Aurora light theme with model performance cards and interactive charts]

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/legomaheggoz-source/SepsisPulse.git
cd SepsisPulse

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Features

- **Dashboard**: Real-time overview of model performance metrics
- **Model Comparison**: Side-by-side analysis of qSOFA, XGBoost-TS, and TFT-Lite predictions
- **Patient Explorer**: Individual patient vital signs and risk assessments
- **Configuration**: Adjustable prediction thresholds and clinical parameters
- **Documentation**: Comprehensive guides on models, metrics, and data formats
- **Aurora Theme**: Modern light interface with clinical color scheme
- **HuggingFace Compatible**: Runs on free tier (2GB RAM limit)

## Model Comparison

**Trained on PhysioNet Challenge 2019 (40,311 patients, 5-fold CV)**

| Metric | qSOFA | XGBoost-TS | TFT-Lite |
|--------|-------|-----------|----------|
| **Type** | Rule-based Heuristic | Gradient Boosting | Deep Learning (Attention) |
| **AUC-ROC** | 0.72 | **0.81** | 0.82 |
| **Utility Score** | 0.31 | **0.70** | 0.68 |
| **Lead Time (hours)** | 4.1 | 5.8 | **6.2** |
| **Sensitivity** | 0.65 | 0.57 | 0.72 |
| **Specificity** | 0.78 | **0.85** | 0.85 |
| **Interpretability** | High | Medium | Low |
| **Inference Speed** | Very Fast | Fast | Moderate |

*XGBoost-TS achieves the best Clinical Utility Score (0.70) - the primary metric for this task.*

## Trained Models

The app includes pre-trained model weights:

```
models/
├── xgboost_ts/weights/
│   └── xgb_sepsis_v1.json      # 429 engineered features
└── tft_lite/weights/
    └── tft_lite_v1.pt          # 149K parameters
```

**Training Details:**
- Dataset: PhysioNet Challenge 2019 (free, open access)
- Patients: 40,311 ICU patients, 7.3% sepsis rate
- Validation: 5-fold stratified patient-level cross-validation
- Primary Metric: Clinical Utility Score (rewards early detection)

See [Training Guide](docs/TRAINING.md) and [Model Integration](docs/MODEL_INTEGRATION.md) for details.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - Rapid Python web apps
- **ML Models**:
  - [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
  - [PyTorch](https://pytorch.org/) - Temporal Fusion Transformer
  - [Scikit-learn](https://scikit-learn.org/) - Utilities & preprocessing
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- **Visualization**: [Plotly](https://plotly.com/), [Altair](https://altair-viz.github.io/)
- **Data Source**: [PhysioNet 2019 Challenge](https://physionet.org/content/challenge-2019/)
- **Deployment**: [HuggingFace Spaces](https://huggingface.co/spaces)

## Project Structure

```
SepsisPulse/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── data/
│   ├── sample/                # Sample dataset subset
│   └── physionet/             # Full PhysioNet dataset (40K patients)
├── models/
│   ├── qsofa/                 # qSOFA baseline model
│   ├── xgboost_ts/            # XGBoost time-series model
│   │   └── weights/           # Trained model weights
│   └── tft_lite/              # Lightweight TFT model
│       └── weights/           # Trained model weights
├── src/
│   ├── data/                  # Data loading & preprocessing
│   ├── evaluation/            # Metrics (utility, lead time)
│   ├── services/              # Model integration services
│   ├── visualization/         # Streamlit components & charts
│   └── utils/                 # Configuration, caching
├── training/                  # Training infrastructure
│   ├── config/                # Training configurations
│   ├── trainers/              # XGBoost & TFT trainers
│   └── scripts/               # Training CLI scripts
├── docs/                      # Documentation
│   ├── USER_GUIDE.md
│   ├── TRAINING.md
│   └── MODEL_INTEGRATION.md
└── tests/                     # Unit tests
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to your branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style

- Python: [Black](https://black.readthedocs.io/) (line length: 88)
- Imports: [isort](https://pycqa.github.io/isort/)
- Type hints: [mypy](http://mypy-lang.org/)

### Testing

```bash
pytest tests/ --cov=src
```

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

Copyright 2026 SepsisPulse Contributors

## Acknowledgments

- **PhysioNet**: [PhysioNet 2019 Challenge](https://physionet.org/content/challenge-2019/) for the sepsis dataset and clinical utility metric
- **Anthropic**: Claude AI for code generation and architectural guidance
- **Sepsis-3 Consensus**: [Singer et al., 2016](https://jamanetwork.com/journals/jama/fullarticle/2526284) for the qSOFA criteria
- **Research Community**: Open-source ML and healthcare AI pioneers

---

**Questions?** Open an [issue](https://github.com/legomaheggoz-source/SepsisPulse/issues) or reach out to the maintainers.

**Version**: 1.0.0 | **Last Updated**: January 2026 | **Models**: Trained on PhysioNet 2019
