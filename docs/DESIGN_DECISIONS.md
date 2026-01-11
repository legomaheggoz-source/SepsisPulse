# SepsisPulse - Design Decisions

This document explains the key technical decisions made during SepsisPulse development, including the reasoning behind each choice.

---

## Table of Contents

1. [Why Streamlit](#why-streamlit)
2. [Why TFT-Lite](#why-tft-lite)
3. [Why Pre-bundled Subset](#why-pre-bundled-subset)
4. [Why Aurora Theme](#why-aurora-theme)
5. [Why Pre-trained Weights Only](#why-pre-trained-weights-only)
6. [Additional Design Decisions](#additional-design-decisions)

---

## Why Streamlit

### Decision

Use Streamlit as the web framework instead of Flask, FastAPI, Django, or other alternatives.

### Alternatives Considered

| Framework | Pros | Cons |
|-----------|------|------|
| **Flask** | Mature, flexible, large ecosystem | Requires frontend development, more boilerplate |
| **FastAPI** | Modern, fast, async support | API-focused, needs separate frontend |
| **Django** | Full-featured, ORM included | Heavyweight for this use case |
| **Dash** | Plotly integration, reactive | Learning curve, less intuitive than Streamlit |
| **Gradio** | ML-focused, simple interface | Limited customization, less polished UI |
| **Streamlit** | Rapid prototyping, Python-only, built-in widgets | Less customizable, unique execution model |

### Reasoning

1. **Rapid Development**: Streamlit allows building interactive dashboards with pure Python. No HTML, CSS, or JavaScript required for basic functionality. This accelerated development significantly.

2. **Data Science Focus**: Streamlit was built for data scientists and ML engineers. It has first-class support for DataFrames, charts, and ML model integration.

3. **Built-in Caching**: The `@st.cache_data` and `@st.cache_resource` decorators provide easy performance optimization without setting up Redis or other caching infrastructure.

4. **Native Plotly Support**: Streamlit renders Plotly figures natively with `st.plotly_chart()`, providing interactive visualizations without configuration.

5. **Deployment Options**: Streamlit apps can be deployed to:
   - Streamlit Community Cloud (free)
   - HuggingFace Spaces (free tier available)
   - Any cloud provider (Heroku, AWS, GCP, Azure)

6. **Community and Ecosystem**: Large community, extensive documentation, and many example projects to reference.

### Trade-offs Accepted

- **Limited Customization**: Streamlit's component library is less flexible than raw HTML/CSS. We mitigated this with custom CSS injection via `st.markdown()`.

- **Execution Model**: Streamlit reruns the entire script on each interaction. This required careful use of caching and session state.

- **Production Readiness**: Streamlit is less battle-tested than Flask/Django for high-traffic production apps. Acceptable for our clinical audit tool use case.

### Code Example

```python
# Streamlit makes this simple
import streamlit as st
import plotly.express as px

st.title("Model Comparison")
model = st.selectbox("Select Model", ["qSOFA", "XGBoost-TS", "TFT-Lite"])
fig = px.line(df, x="time", y="prediction", title=f"{model} Predictions")
st.plotly_chart(fig)

# vs Flask (would require HTML template + JavaScript)
```

---

## Why TFT-Lite

### Decision

Implement a lightweight version of the Temporal Fusion Transformer (TFT) rather than using the full TFT architecture or other deep learning models.

### Alternatives Considered

| Model | Parameters | Memory | Accuracy |
|-------|------------|--------|----------|
| **Full TFT** | ~10M | ~4GB | Highest |
| **LSTM** | ~2M | ~800MB | Good |
| **Transformer** | ~5M | ~2GB | Very Good |
| **CNN-LSTM** | ~3M | ~1.2GB | Good |
| **TFT-Lite** | ~500K | ~500MB | Very Good |

### Reasoning

1. **2GB RAM Constraint**: The primary deployment target is HuggingFace Spaces free tier, which has a 2GB RAM limit. The full TFT would not fit.

2. **Architecture Preservation**: TFT-Lite preserves the key architectural innovations of TFT:
   - Variable Selection Networks (feature importance)
   - Gated Residual Networks (non-linear processing)
   - Multi-head Attention (temporal dependencies)
   - Interpretability features (attention weights)

3. **Minimal Accuracy Loss**: Based on literature, reducing hidden dimensions and layers causes moderate accuracy degradation:
   - Hidden 256 -> 32: ~3-5% AUC drop
   - 2 LSTM layers -> 1: ~1-2% AUC drop
   - 4 attention heads -> 2: ~1% AUC drop
   - Combined: ~5-8% degradation, still competitive with XGBoost

4. **Inference Speed**: Smaller model means faster inference, important for real-time clinical use.

5. **Interpretability**: TFT's attention mechanisms provide interpretability that simpler models like LSTM lack. This is valuable for clinical acceptance.

### Architecture Comparison

```
Original TFT:
- hidden_size: 256
- lstm_layers: 2
- attention_heads: 4
- max_seq_length: 168 (1 week)
- parameters: ~10M
- memory: ~4GB

TFT-Lite:
- hidden_size: 32
- lstm_layers: 1
- attention_heads: 2
- max_seq_length: 24 (1 day)
- parameters: ~500K
- memory: ~500MB
```

### Trade-offs Accepted

- **Reduced Accuracy**: Approximately 5-8% lower than full TFT. Mitigated by the architecture still being competitive with traditional ML.

- **Shorter Sequence Length**: Limited to 24 hours instead of 168 hours. This is sufficient for most sepsis prediction scenarios since onset typically occurs within 24-48 hours.

- **Training Complexity**: TFT-Lite still requires more data and compute to train than XGBoost. Mitigated by providing pre-trained weights.

---

## Why Pre-bundled Subset

### Decision

Include a synthetic sample dataset (20 patients) rather than requiring users to download the full PhysioNet dataset.

### The PhysioNet Access Problem

1. **Registration Required**: PhysioNet datasets require creating an account and agreeing to a data use agreement.

2. **Approval Delay**: Some datasets require credentialing, which can take days to weeks.

3. **Large Download**: The full PhysioNet 2019 dataset is several GB.

4. **User Friction**: Requiring data download before using the app would significantly reduce adoption.

### Reasoning

1. **Immediate Usability**: Users can explore the dashboard immediately without any setup.

2. **Demo Capability**: The sample data enables demonstrations and presentations without real patient data.

3. **Development and Testing**: Developers can run tests and debug without the full dataset.

4. **Realistic Characteristics**: The synthetic data mimics key characteristics of the real dataset:
   - 41 columns in correct format
   - Realistic vital sign ranges
   - Appropriate missing value patterns (>80% for labs)
   - Sepsis progression patterns

5. **No Privacy Concerns**: Synthetic data avoids any HIPAA or data privacy issues.

### Sample Data Characteristics

| Characteristic | Sample Data | Real Data |
|----------------|-------------|-----------|
| Patients | 20 | 40,336 |
| Sepsis Cases | 10 (50%) | ~7% |
| Hours per Patient | 24-72 | 8-336 |
| Missing Rate (Vitals) | ~10% | ~10% |
| Missing Rate (Labs) | ~80% | ~80-95% |

### Trade-offs Accepted

- **Limited Evaluation**: Statistical evaluation on 20 patients is not meaningful. Users must download full data for real analysis.

- **Class Balance**: Sample is 50% sepsis (vs ~7% in reality). Noted in documentation.

- **Synthetic Patterns**: Generated data cannot capture all correlations and patterns in real clinical data.

---

## Why Aurora Theme

### Decision

Use a custom "Aurora" dark theme with solar-inspired colors instead of Streamlit's default light theme.

### Design Goals

1. **Reduce Eye Strain**: Clinical users may view dashboards for extended periods. Dark themes cause less eye fatigue.

2. **Modern Aesthetic**: Dark themes convey a modern, professional appearance appropriate for clinical decision support.

3. **Data Visualization**: Dark backgrounds make colorful charts and visualizations pop more prominently.

4. **Differentiation**: A custom theme distinguishes SepsisPulse from generic Streamlit apps.

### Color Palette

```python
COLORS = {
    "background": "#0d1117",      # Deep space black
    "card_bg": "#161b22",         # Slightly lighter cards
    "card_border": "#30363d",     # Subtle borders
    "primary": "#58a6ff",         # Cyan blue (primary actions)
    "secondary": "#8b949e",       # Gray (secondary text)
    "success": "#3fb950",         # Green (positive/success)
    "warning": "#d29922",         # Amber (warnings/caution)
    "danger": "#f85149",          # Red (errors/critical)
    "text_primary": "#f0f6fc",    # Near-white text
    "text_secondary": "#8b949e",  # Gray secondary text
}
```

### Reasoning

1. **GitHub Inspiration**: The color palette is inspired by GitHub's dark theme, which is well-tested for code and data display.

2. **Accessibility**: Colors were chosen to maintain sufficient contrast ratios for accessibility:
   - Text on background: 15.4:1 (exceeds WCAG AAA)
   - Primary on background: 6.2:1 (exceeds WCAG AA)

3. **Clinical Color Conventions**:
   - Red = Critical/Danger (sepsis alert)
   - Amber = Warning (elevated risk)
   - Green = Success/Normal (no sepsis)

4. **Chart Compatibility**: The dark background works well with Plotly's default color sequences.

### Implementation

Theme is applied via:
1. `.streamlit/config.toml` for base Streamlit colors
2. `src/visualization/theme.py` for CSS injection
3. `get_plotly_template()` for consistent chart styling

### Trade-offs Accepted

- **Printing**: Dark themes don't print well. Users needing printed reports should use browser print settings.

- **Light Mode Preference**: Some users prefer light themes. A toggle could be added but increases complexity.

---

## Why Pre-trained Weights Only

### Decision

Provide pre-trained model weights for inference only, without including training code or pipelines.

### Alternatives Considered

1. **Full Training Pipeline**: Include data preprocessing, training loops, hyperparameter tuning
2. **Transfer Learning**: Include fine-tuning capabilities for custom data
3. **AutoML**: Include automated model selection and training
4. **Inference Only**: Provide pre-trained weights, no training

### Reasoning

1. **Scope Management**: Including training would significantly expand the codebase:
   - Data splitting and cross-validation
   - Training loops and checkpointing
   - Hyperparameter optimization
   - GPU support and distributed training
   - Experiment tracking

2. **Compute Requirements**: Training requires:
   - Full PhysioNet dataset (users may not have access)
   - Significant compute (hours for XGBoost, days for TFT)
   - GPU for TFT training (not available on free tiers)

3. **Reproducibility**: Pre-trained weights ensure consistent results across users. Training could produce varying results due to random initialization, data ordering, etc.

4. **Use Case Focus**: SepsisPulse is a clinical utility auditor, not a training framework. The goal is to compare prediction approaches, not to train new models.

5. **Target Audience**: Primary users are clinical informaticists and researchers evaluating models, not ML engineers training them.

### What's Included

```
models/
+-- qsofa/
|   +-- qsofa_model.py       # Rule-based, no weights needed
+-- xgboost_ts/
|   +-- xgboost_model.py     # Wrapper with weight loading
|   +-- weights/
|       +-- xgb_sepsis_v1.json  # Pre-trained weights
+-- tft_lite/
    +-- architecture.py      # Model definition
    +-- tft_model.py         # Wrapper with weight loading
    +-- weights/
        +-- tft_lite_v1.pt   # Pre-trained weights
```

### Trade-offs Accepted

- **No Custom Training**: Users cannot train on their own data without additional code.

- **Fixed Architecture**: Users cannot experiment with different hyperparameters.

- **Model Staleness**: Weights may become outdated as new data or techniques emerge.

### Mitigation

- Documentation explains how to train models externally
- Model wrappers support loading custom weights
- Architecture code can be extended for training

---

## Additional Design Decisions

### Decision: Forward-Fill Imputation Strategy

**Reasoning**: In clinical settings, lab values remain valid until re-measured. Forward-fill is clinically appropriate because a patient's creatinine level from 4 hours ago is still informative if not re-measured.

**Alternative**: Mean imputation or model-based imputation. These ignore the temporal structure of the data.

---

### Decision: Clinical Utility as Primary Metric

**Reasoning**: Traditional ML metrics (AUC-ROC, F1) don't capture clinical priorities. The PhysioNet utility function explicitly rewards early detection and penalizes late predictions and false alarms, aligning with clinical goals.

**Alternative**: Optimize for AUC-ROC or F1. These would not differentiate between a prediction 12 hours early vs 1 hour late.

---

### Decision: 24-Hour Sequence Length for TFT-Lite

**Reasoning**: Sepsis onset in the PhysioNet data typically occurs within the first 24-48 hours. Longer sequences (168h) provide diminishing returns while significantly increasing memory usage.

**Alternative**: Variable sequence lengths with padding/masking. More complex to implement and may not improve performance.

---

### Decision: Demo Mode for Missing Weights

**Reasoning**: Rather than crashing when weights are unavailable, models fall back to demo mode with reproducible pseudo-random predictions. This allows the UI to function for demonstration purposes.

**Alternative**: Raise exceptions for missing weights. This would break the demo experience for new users.

---

### Decision: Singleton Configuration Pattern

**Reasoning**: Using `get_config()` ensures all modules access the same configuration instance, preventing inconsistencies from multiple Config objects with different settings.

**Alternative**: Pass config explicitly to every function. More verbose and error-prone.

---

### Decision: Dictionary-based Patient Data Structure

**Reasoning**: Using `Dict[patient_id, DataFrame]` allows easy patient-level operations and preserves patient identity throughout the pipeline.

**Alternative**: Single concatenated DataFrame with patient_id column. Harder to process patients individually and risks mixing patient data.

---

## Summary

These design decisions prioritize:

1. **Accessibility**: Easy setup, immediate usability, free deployment
2. **Clinical Relevance**: Utility score, lead time, interpretability
3. **Resource Efficiency**: 2GB RAM target, inference-only
4. **Developer Experience**: Python-only, minimal configuration
5. **Maintainability**: Clear architecture, documented decisions
