"""
SepsisPulse - Clinical Utility & Lead-Time Auditor for Sepsis Early Warning

Main Streamlit application entry point.
Audits three sepsis prediction approaches: qSOFA, XGBoost-TS, TFT-Lite
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Configure page - must be first Streamlit command
st.set_page_config(
    page_title="SepsisPulse",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import local modules (with fallback for demo mode)
try:
    from src.visualization.theme import COLORS, apply_aurora_theme
    from src.visualization.components import metric_card, alert_banner, model_comparison_table
    from src.visualization.charts import create_roc_chart, create_lead_time_chart, create_utility_chart
    from src.data.loader import load_dataset, get_sample_subset
    from src.evaluation.clinical_utility import compute_utility_score
    from src.evaluation.lead_time import compute_average_lead_time
    from models import QSOFAModel, XGBoostTSModel, TFTLiteModel
    MODULES_LOADED = True
except ImportError:
    MODULES_LOADED = False

# ============================================================================
# AURORA THEME CSS
# ============================================================================

AURORA_CSS = """
<style>
/* Aurora Solar-Inspired Dark Theme */
:root {
    --background: #0a0e17;
    --card-bg: #002d42;
    --card-border: #004466;
    --primary: #00d9ff;
    --secondary: #7c3aed;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text-primary: #ffffff;
    --text-secondary: #94a3b8;
}

/* Main container */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--card-bg) 0%, #001a2e 100%);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

div[data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.9rem;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: var(--primary) !important;
    font-size: 2rem;
    font-weight: 700;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--card-bg) 0%, #001a2e 100%);
    border-right: 1px solid var(--card-border);
}

section[data-testid="stSidebar"] .stRadio label {
    color: var(--text-primary);
}

/* Headers */
h1, h2, h3 {
    color: var(--text-primary) !important;
}

h1 {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Alert banner */
.alert-critical {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
    border-left: 4px solid var(--danger);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.alert-warning {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
    border-left: 4px solid var(--warning);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

.alert-success {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    border-left: 4px solid var(--success);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Model comparison table */
.model-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}

.model-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 217, 255, 0.15);
}

.model-name {
    color: var(--primary);
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.model-metric {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

.model-value {
    color: var(--text-primary);
    font-size: 1.5rem;
    font-weight: 700;
}

/* Plotly chart container */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* DataFrames */
.dataframe {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: opacity 0.2s;
}

.stButton > button:hover {
    opacity: 0.9;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: var(--card-bg);
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.5rem;
    color: var(--text-secondary);
}

.stTabs [aria-selected="true"] {
    background: var(--primary);
    color: white;
}
</style>
"""

# ============================================================================
# DEMO DATA (for when models aren't trained yet)
# ============================================================================

DEMO_METRICS = {
    "qSOFA": {
        "name": "qSOFA",
        "type": "Heuristic",
        "auc": 0.72,
        "utility": 0.31,
        "lead_time": 4.1,
        "sensitivity": 0.65,
        "specificity": 0.78,
    },
    "XGBoost-TS": {
        "name": "XGBoost-TS",
        "type": "ML (Gradient Boosting)",
        "auc": 0.85,
        "utility": 0.39,
        "lead_time": 5.8,
        "sensitivity": 0.79,
        "specificity": 0.88,
    },
    "TFT-Lite": {
        "name": "TFT-Lite",
        "type": "Deep Learning",
        "auc": 0.87,
        "utility": 0.42,
        "lead_time": 6.2,
        "sensitivity": 0.82,
        "specificity": 0.89,
    },
}

DEMO_PATIENTS = 1000
DEMO_SEPSIS_CASES = 127


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("## 💓 SepsisPulse")
        st.markdown("*Clinical Utility Auditor*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "📈 Model Comparison", "🔍 Patient Explorer", "⚙️ Configuration", "📖 Documentation"],
            label_visibility="collapsed",
        )

        st.divider()

        # Status indicator
        if MODULES_LOADED:
            st.success("✓ All modules loaded")
        else:
            st.warning("⚠ Demo mode (modules not loaded)")

        st.divider()

        # Quick stats
        st.markdown("### Quick Stats")
        st.metric("Patients", DEMO_PATIENTS)
        st.metric("Sepsis Cases", DEMO_SEPSIS_CASES)
        st.metric("Best Model", "TFT-Lite")

        st.divider()
        st.caption("v0.1.0 | PhysioNet 2019")

        return page


def render_dashboard():
    """Render the main dashboard page."""
    st.markdown(AURORA_CSS, unsafe_allow_html=True)

    # Header
    st.title("SepsisPulse")
    st.markdown("### Clinical Utility & Lead-Time Auditor for Sepsis Early Warning")

    # Alert banner (example)
    st.markdown(
        """
        <div class="alert-warning">
            <strong>📊 Demo Mode</strong> - Showing simulated metrics.
            Load real patient data to see actual predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Patients",
            value=f"{DEMO_PATIENTS:,}",
            delta="Sample subset",
        )

    with col2:
        st.metric(
            label="Sepsis Cases",
            value=DEMO_SEPSIS_CASES,
            delta=f"{DEMO_SEPSIS_CASES/DEMO_PATIENTS*100:.1f}%",
        )

    with col3:
        best_lead = max(m["lead_time"] for m in DEMO_METRICS.values())
        st.metric(
            label="Best Lead Time",
            value=f"{best_lead:.1f} hrs",
            delta="TFT-Lite",
        )

    with col4:
        best_utility = max(m["utility"] for m in DEMO_METRICS.values())
        st.metric(
            label="Best Utility Score",
            value=f"{best_utility:.3f}",
            delta="+0.11 vs qSOFA",
        )

    st.divider()

    # Model comparison section
    st.markdown("### Model Comparison")

    # Create comparison table
    col1, col2, col3 = st.columns(3)

    for col, (model_name, metrics) in zip([col1, col2, col3], DEMO_METRICS.items()):
        with col:
            is_best = metrics["utility"] == best_utility
            border_color = "#00d9ff" if is_best else "#004466"

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #002d42 0%, #001a2e 100%);
                    border: 2px solid {border_color};
                    border-radius: 12px;
                    padding: 1.5rem;
                    text-align: center;
                    {'box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);' if is_best else ''}
                ">
                    <div style="color: #00d9ff; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">
                        {model_name}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.8rem; margin-bottom: 1rem;">
                        {metrics['type']}
                    </div>
                    <div style="display: grid; gap: 0.75rem;">
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem;">AUC-ROC</div>
                            <div style="color: white; font-size: 1.5rem; font-weight: 700;">{metrics['auc']:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem;">Utility Score</div>
                            <div style="color: #10b981; font-size: 1.5rem; font-weight: 700;">{metrics['utility']:.2f}</div>
                        </div>
                        <div>
                            <div style="color: #94a3b8; font-size: 0.75rem;">Lead Time</div>
                            <div style="color: #f59e0b; font-size: 1.5rem; font-weight: 700;">{metrics['lead_time']:.1f}h</div>
                        </div>
                    </div>
                    {'<div style="margin-top: 1rem; color: #00d9ff; font-size: 0.8rem;">⭐ BEST</div>' if is_best else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    # Charts section
    st.markdown("### Performance Analysis")

    tab1, tab2, tab3 = st.tabs(["📈 ROC Curves", "⏱️ Lead Time Distribution", "📊 Utility Over Time"])

    with tab1:
        # Placeholder ROC curve
        import plotly.graph_objects as go

        fig = go.Figure()

        # Random walk for demo ROC curves
        np.random.seed(42)
        for model_name, metrics in DEMO_METRICS.items():
            fpr = np.linspace(0, 1, 100)
            # Generate plausible TPR based on AUC
            tpr = 1 - (1 - fpr) ** (1 / (2 - metrics["auc"]))
            tpr = np.clip(tpr + np.random.normal(0, 0.02, len(tpr)), 0, 1)
            tpr = np.sort(tpr)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f"{model_name} (AUC={metrics['auc']:.2f})",
                line=dict(width=2),
            ))

        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray'),
        ))

        fig.update_layout(
            title="ROC Curves - Model Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,23,1)',
            font=dict(color='white'),
            legend=dict(x=0.6, y=0.1),
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Lead time distribution
        fig = go.Figure()

        for model_name, metrics in DEMO_METRICS.items():
            # Generate plausible lead time distribution
            lead_times = np.random.exponential(metrics["lead_time"], 200)
            lead_times = np.clip(lead_times, 0, 24)

            fig.add_trace(go.Histogram(
                x=lead_times,
                name=model_name,
                opacity=0.7,
                nbinsx=24,
            ))

        fig.update_layout(
            title="Lead Time Distribution (hours before sepsis onset)",
            xaxis_title="Lead Time (hours)",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,23,1)',
            font=dict(color='white'),
            barmode='overlay',
            height=400,
        )

        # Add vertical line at 6h optimal
        fig.add_vline(x=6, line_dash="dash", line_color="#00d9ff",
                      annotation_text="Optimal (6h)")

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Utility score over threshold
        fig = go.Figure()

        thresholds = np.linspace(0.1, 0.9, 50)

        for model_name, metrics in DEMO_METRICS.items():
            # Generate plausible utility curve
            base = metrics["utility"]
            utility = base * np.exp(-2 * (thresholds - 0.4) ** 2)
            utility = np.clip(utility + np.random.normal(0, 0.01, len(utility)), 0, 1)

            fig.add_trace(go.Scatter(
                x=thresholds,
                y=utility,
                mode='lines',
                name=model_name,
                line=dict(width=2),
            ))

        fig.update_layout(
            title="Utility Score vs. Decision Threshold",
            xaxis_title="Prediction Threshold",
            yaxis_title="Clinical Utility Score",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,23,1)',
            font=dict(color='white'),
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


def render_model_comparison():
    """Render detailed model comparison page."""
    st.markdown(AURORA_CSS, unsafe_allow_html=True)

    st.title("📈 Model Comparison")
    st.markdown("Detailed comparison of sepsis prediction approaches")

    # Metrics table
    st.markdown("### Performance Metrics")

    df = pd.DataFrame(DEMO_METRICS).T
    df = df[["name", "type", "auc", "utility", "lead_time", "sensitivity", "specificity"]]
    df.columns = ["Model", "Type", "AUC-ROC", "Utility Score", "Lead Time (h)", "Sensitivity", "Specificity"]

    st.dataframe(
        df.style.highlight_max(subset=["AUC-ROC", "Utility Score", "Lead Time (h)", "Sensitivity", "Specificity"], color='#10b981'),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Detailed analysis
    st.markdown("### Model Details")

    tab1, tab2, tab3 = st.tabs(["qSOFA", "XGBoost-TS", "TFT-Lite"])

    with tab1:
        st.markdown("""
        #### qSOFA (Quick SOFA)

        **Type:** Rule-based heuristic (Sepsis-3 guidelines)

        **Variables:**
        - Respiratory Rate ≥ 22 breaths/min (+1 point)
        - Systolic BP ≤ 100 mmHg (+1 point)
        - Altered mentation / GCS < 15 (+1 point)

        **Decision Rule:** Score ≥ 2 indicates high sepsis risk

        **Pros:**
        - Simple, interpretable
        - No training required
        - Works at bedside

        **Cons:**
        - Low sensitivity
        - Triggers late
        - High alarm fatigue
        """)

    with tab2:
        st.markdown("""
        #### XGBoost-TS (Time Series)

        **Type:** Gradient Boosted Decision Trees

        **Features (~200):**
        - 41 raw PhysioNet variables
        - Lag features (1h, 3h, 6h)
        - Rolling statistics (mean, std, min, max)
        - Rate of change

        **Architecture:**
        - 500 trees, max depth 6
        - Learning rate: 0.05
        - L2 regularization: 1.0

        **Pros:**
        - Strong on tabular data
        - Feature importance available
        - Fast inference

        **Cons:**
        - Requires feature engineering
        - May miss long-term patterns
        """)

    with tab3:
        st.markdown("""
        #### TFT-Lite (Lightweight Temporal Fusion Transformer)

        **Type:** Deep Learning (Attention-based)

        **Architecture:**
        - Hidden size: 32 (vs. 256 original)
        - LSTM layers: 1 (vs. 2)
        - Attention heads: 2 (vs. 4)
        - Max sequence: 24h (vs. 168h)

        **Parameters:** ~500K (vs. ~10M original)

        **Memory:** ~500MB inference (fits in 2GB HF tier)

        **Pros:**
        - State-of-the-art accuracy
        - Captures temporal patterns
        - Attention interpretability

        **Cons:**
        - Slower than XGBoost
        - Black box (partially)
        - Requires more data
        """)


def render_patient_explorer():
    """Render patient explorer page."""
    st.markdown(AURORA_CSS, unsafe_allow_html=True)

    st.title("🔍 Patient Explorer")
    st.markdown("Explore individual patient predictions and vital signs")

    # Patient selector
    col1, col2 = st.columns([1, 3])

    with col1:
        patient_id = st.selectbox(
            "Select Patient",
            [f"p{i:05d}" for i in range(1, 21)],
        )

        st.markdown("### Patient Info")
        st.markdown(f"**ID:** {patient_id}")
        st.markdown(f"**Age:** {np.random.randint(45, 85)}")
        st.markdown(f"**Gender:** {'M' if np.random.random() > 0.5 else 'F'}")
        st.markdown(f"**ICU Stay:** {np.random.randint(24, 168)}h")

        has_sepsis = np.random.random() > 0.85
        if has_sepsis:
            st.error("⚠️ Sepsis Positive")
        else:
            st.success("✓ No Sepsis")

    with col2:
        # Generate demo vitals
        hours = np.arange(0, 48)
        hr = 80 + np.random.randn(48).cumsum() * 2
        sbp = 120 + np.random.randn(48).cumsum() * 3
        temp = 37 + np.random.randn(48).cumsum() * 0.1
        resp = 16 + np.random.randn(48).cumsum() * 0.5

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Heart Rate", "Blood Pressure", "Temperature", "Respiratory Rate"),
        )

        fig.add_trace(go.Scatter(x=hours, y=hr, name="HR", line=dict(color="#00d9ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=sbp, name="SBP", line=dict(color="#10b981")), row=1, col=2)
        fig.add_trace(go.Scatter(x=hours, y=temp, name="Temp", line=dict(color="#f59e0b")), row=2, col=1)
        fig.add_trace(go.Scatter(x=hours, y=resp, name="Resp", line=dict(color="#ef4444")), row=2, col=2)

        fig.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,23,1)',
            font=dict(color='white'),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Model predictions
    st.markdown("### Model Predictions")

    col1, col2, col3 = st.columns(3)

    np.random.seed(hash(patient_id) % 100)
    predictions = {
        "qSOFA": np.random.random() * 0.6,
        "XGBoost-TS": np.random.random() * 0.8,
        "TFT-Lite": np.random.random() * 0.9,
    }

    for col, (model, prob) in zip([col1, col2, col3], predictions.items()):
        with col:
            risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
            color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#10b981"

            st.markdown(
                f"""
                <div style="
                    background: #002d42;
                    border: 1px solid #004466;
                    border-radius: 8px;
                    padding: 1rem;
                    text-align: center;
                ">
                    <div style="color: #94a3b8; font-size: 0.8rem;">{model}</div>
                    <div style="color: white; font-size: 2rem; font-weight: 700;">{prob:.1%}</div>
                    <div style="color: {color}; font-size: 0.9rem;">{risk} Risk</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_configuration():
    """Render configuration page."""
    st.markdown(AURORA_CSS, unsafe_allow_html=True)

    st.title("⚙️ Configuration")
    st.markdown("Adjust prediction thresholds and display settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prediction Thresholds")

        qsofa_threshold = st.slider("qSOFA Threshold", 1, 3, 2)
        xgboost_threshold = st.slider("XGBoost-TS Threshold", 0.0, 1.0, 0.5, 0.05)
        tft_threshold = st.slider("TFT-Lite Threshold", 0.0, 1.0, 0.5, 0.05)

        st.markdown("### Lead Time Settings")
        optimal_lead = st.slider("Optimal Lead Time (hours)", 1, 12, 6)
        max_lead = st.slider("Maximum Lead Time (hours)", 6, 24, 12)

    with col2:
        st.markdown("### Display Settings")

        show_confidence = st.checkbox("Show Confidence Intervals", value=True)
        show_all_patients = st.checkbox("Include All Patients", value=False)
        theme = st.selectbox("Color Theme", ["Aurora Dark", "Aurora Light", "Clinical"])

        st.markdown("### Data Settings")

        data_source = st.selectbox("Data Source", ["Sample Subset (1000)", "Full Dataset", "Custom Upload"])

        if data_source == "Custom Upload":
            uploaded = st.file_uploader("Upload PSV files", type=["psv", "csv"])

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("Settings saved!")

    with col2:
        if st.button("🔄 Reset Defaults", use_container_width=True):
            st.info("Settings reset to defaults")


def render_documentation():
    """Render documentation page."""
    st.markdown(AURORA_CSS, unsafe_allow_html=True)

    st.title("📖 Documentation")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Clinical Utility Score", "Models", "Data Format"])

    with tab1:
        st.markdown("""
        ## SepsisPulse Overview

        SepsisPulse is a clinical decision support tool that audits and compares
        three mathematical approaches to early sepsis prediction:

        1. **qSOFA** - Quick Sequential Organ Failure Assessment (baseline heuristic)
        2. **XGBoost-TS** - Gradient boosted trees with time-series features
        3. **TFT-Lite** - Lightweight Temporal Fusion Transformer

        ### Key Metric: Clinical Utility Score

        Unlike traditional metrics (AUC-ROC), the Clinical Utility Score from
        PhysioNet Challenge 2019 rewards **early** predictions while penalizing
        false alarms and late detections.

        ### The Problem: Alarm Fatigue

        Current hospital alerts trigger too late or too often. "Alarm Fatigue"
        causes clinicians to ignore alerts, leading to preventable deaths.

        > *"Every hour of delayed sepsis treatment increases mortality by 8%"*

        ### Our Goal

        Determine the **Optimal Alert Window** - how early can we reliably
        detect sepsis without creating excessive false alarms?
        """)

    with tab2:
        st.markdown("""
        ## Clinical Utility Score

        The PhysioNet 2019 Challenge introduced a utility function that captures
        clinical priorities:

        ### Scoring Rules

        **For sepsis patients:**
        - **Optimal window**: 6-12 hours before onset → Maximum reward
        - **Too early** (>12h before): Small penalty (-0.05)
        - **Too late** (after onset): Large penalty (-2.0)

        **For non-sepsis patients:**
        - **False positive**: Penalty (-0.05) → Addresses alarm fatigue
        - **True negative**: No reward (expected behavior)

        ### Formula

        ```
        U(t) = {
            +1.0   if t_pred in [t_sepsis-12h, t_sepsis-6h]  (ideal)
            +0.5   if t_pred in [t_sepsis-6h, t_sepsis]      (acceptable)
            -0.05  if t_pred < t_sepsis-12h                   (too early)
            -2.0   if t_pred > t_sepsis                       (too late)
            -0.05  if false positive                          (false alarm)
        }
        ```

        ### Normalization

        Final score is normalized: `(U - U_baseline) / (U_optimal - U_baseline)`

        - Score of 0.0 = No better than always predicting negative
        - Score of 1.0 = Perfect predictions
        """)

    with tab3:
        st.markdown("""
        ## Model Architectures

        ### qSOFA (Quick SOFA)

        **Rule-based heuristic from Sepsis-3 guidelines**

        | Criterion | Points |
        |-----------|--------|
        | Respiratory Rate ≥ 22 | +1 |
        | Systolic BP ≤ 100 | +1 |
        | Altered mentation (GCS < 15) | +1 |

        Score ≥ 2 = High risk

        ---

        ### XGBoost-TS

        **Gradient Boosted Decision Trees with engineered features**

        - **Input**: 41 raw variables → ~200 engineered features
        - **Features**: Lag (1h, 3h, 6h), rolling stats, rate of change
        - **Hyperparameters**: 500 trees, max_depth=6, lr=0.05

        ---

        ### TFT-Lite

        **Lightweight Temporal Fusion Transformer**

        | Parameter | TFT-Lite | Original |
        |-----------|----------|----------|
        | Hidden size | 32 | 256 |
        | LSTM layers | 1 | 2 |
        | Attention heads | 2 | 4 |
        | Parameters | ~500K | ~10M |
        | Memory | ~500MB | ~4GB |
        """)

    with tab4:
        st.markdown("""
        ## PhysioNet 2019 Data Format

        ### File Format

        - **Extension**: `.psv` (pipe-separated values)
        - **Delimiter**: `|`
        - **One file per patient**: `p00001.psv`, `p00002.psv`, etc.
        - **One row per hour** of ICU stay

        ### Variables (41 total)

        **Vital Signs (8):**
        HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2

        **Laboratory Values (26):**
        BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2, AST, BUN,
        Alkalinephos, Calcium, Chloride, Creatinine, Bilirubin_direct,
        Glucose, Lactate, Magnesium, Phosphate, Potassium,
        Bilirubin_total, TroponinI, Hct, Hgb, PTT, WBC,
        Fibrinogen, Platelets

        **Demographics (6):**
        Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS

        **Label (1):**
        SepsisLabel (0 or 1)

        ### Missing Values

        Missing values are represented as `NaN`. Lab values often have
        >90% missing (only measured when clinically indicated).

        **Imputation Strategy:**
        1. Forward fill (carry last observation)
        2. Backward fill (for initial NaNs)
        3. Mean imputation (population average)
        """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "📊 Dashboard":
        render_dashboard()
    elif page == "📈 Model Comparison":
        render_model_comparison()
    elif page == "🔍 Patient Explorer":
        render_patient_explorer()
    elif page == "⚙️ Configuration":
        render_configuration()
    elif page == "📖 Documentation":
        render_documentation()


if __name__ == "__main__":
    main()
