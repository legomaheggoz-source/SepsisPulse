"""Interactive Plotly charts for SepsisPulse dashboard."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import COLORS, get_plotly_template


def create_roc_chart(
    results: Dict[str, Tuple[float, np.ndarray, np.ndarray]]
) -> go.Figure:
    """
    Create a multi-model ROC curves comparison chart.

    Args:
        results: Dictionary mapping model names to tuples of
                 (AUC score, FPR array, TPR array).

    Returns:
        go.Figure: Plotly figure with ROC curves for all models.
    """
    template = get_plotly_template()
    colorway = template["layout"]["colorway"]

    fig = go.Figure()

    # Add ROC curve for each model
    for idx, (model_name, (auc_score, fpr, tpr)) in enumerate(results.items()):
        color = colorway[idx % len(colorway)]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model_name} (AUC = {auc_score:.3f})",
                line=dict(color=color, width=2),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "FPR: %{x:.3f}<br>"
                    "TPR: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color=COLORS["secondary"], width=1, dash="dash"),
            showlegend=True,
        )
    )

    # Apply Aurora theme layout
    fig.update_layout(
        **template["layout"],
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(**template["layout"]["xaxis"], range=[0, 1]),
        yaxis=dict(**template["layout"]["yaxis"], range=[0, 1]),
        legend=dict(
            **template["layout"]["legend"],
            x=0.98,
            y=0.02,
            xanchor="right",
            yanchor="bottom",
        ),
        hovermode="closest",
    )

    return fig


def create_lead_time_chart(lead_times: Dict[str, np.ndarray]) -> go.Figure:
    """
    Create a lead time distribution histogram per model.

    Args:
        lead_times: Dictionary mapping model names to arrays of
                    lead times (in hours before sepsis onset).

    Returns:
        go.Figure: Plotly figure with overlapping histograms.
    """
    template = get_plotly_template()
    colorway = template["layout"]["colorway"]

    fig = go.Figure()

    # Determine common bin edges
    all_times = np.concatenate(list(lead_times.values()))
    max_time = np.nanmax(all_times[np.isfinite(all_times)])
    bin_edges = np.linspace(0, max_time, 25)

    for idx, (model_name, times) in enumerate(lead_times.items()):
        color = colorway[idx % len(colorway)]

        # Filter out NaN and infinite values
        valid_times = times[np.isfinite(times)]
        mean_lead_time = np.mean(valid_times) if len(valid_times) > 0 else 0

        fig.add_trace(
            go.Histogram(
                x=valid_times,
                name=f"{model_name} (mean: {mean_lead_time:.1f}h)",
                marker_color=color,
                opacity=0.7,
                xbins=dict(
                    start=0,
                    end=max_time,
                    size=(max_time / 24),
                ),
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Lead Time: %{x:.1f}h<br>"
                    "Count: %{y}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Apply Aurora theme layout
    fig.update_layout(
        **template["layout"],
        title="Sepsis Prediction Lead Time Distribution",
        xaxis_title="Lead Time Before Onset (hours)",
        yaxis_title="Number of Predictions",
        barmode="overlay",
        legend=dict(
            **template["layout"]["legend"],
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
        ),
        hovermode="x unified",
    )

    return fig


def create_utility_chart(utilities: Dict[str, float]) -> go.Figure:
    """
    Create a bar chart comparing utility scores across models.

    Args:
        utilities: Dictionary mapping model names to utility scores.

    Returns:
        go.Figure: Plotly figure with horizontal bar chart.
    """
    template = get_plotly_template()
    colorway = template["layout"]["colorway"]

    # Sort by utility score descending
    sorted_models = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
    model_names = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]

    # Assign colors based on score ranking
    colors = [colorway[i % len(colorway)] for i in range(len(model_names))]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=scores,
            y=model_names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color=COLORS["card_border"], width=1),
            ),
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
            textfont=dict(color=COLORS["text_primary"]),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Utility Score: %{x:.4f}<br>"
                "<extra></extra>"
            ),
        )
    )

    # Determine x-axis range
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    x_min = min(0, min_score - abs(min_score) * 0.1)
    x_max = max_score + abs(max_score) * 0.15

    # Apply Aurora theme layout
    fig.update_layout(
        **template["layout"],
        title="Clinical Utility Score Comparison",
        xaxis_title="Utility Score",
        yaxis_title="",
        xaxis=dict(**template["layout"]["xaxis"], range=[x_min, x_max]),
        yaxis=dict(**template["layout"]["yaxis"], automargin=True),
        showlegend=False,
        height=max(300, len(model_names) * 50 + 100),
    )

    return fig


def create_patient_timeline(
    vitals: pd.DataFrame,
    predictions: Dict[str, np.ndarray],
) -> go.Figure:
    """
    Create a patient vital signs timeline with prediction overlay.

    Args:
        vitals: DataFrame with columns including 'time' (or index as time),
                and vital sign columns like 'hr', 'sbp', 'temp', 'resp', 'o2sat'.
        predictions: Dictionary mapping model names to prediction probability arrays
                     aligned with the vitals DataFrame index.

    Returns:
        go.Figure: Plotly figure with subplots for vitals and predictions.
    """
    template = get_plotly_template()
    colorway = template["layout"]["colorway"]

    # Determine time axis
    if "time" in vitals.columns:
        time_axis = vitals["time"]
    else:
        time_axis = vitals.index

    # Define vital signs to plot with their display names and normal ranges
    vital_configs = {
        "hr": {"name": "Heart Rate", "unit": "bpm", "normal": (60, 100)},
        "sbp": {"name": "Systolic BP", "unit": "mmHg", "normal": (90, 140)},
        "temp": {"name": "Temperature", "unit": "\u00b0C", "normal": (36.1, 37.2)},
        "resp": {"name": "Resp Rate", "unit": "/min", "normal": (12, 20)},
        "o2sat": {"name": "SpO2", "unit": "%", "normal": (95, 100)},
    }

    # Filter to available vitals
    available_vitals = [v for v in vital_configs if v in vitals.columns]
    n_vital_rows = len(available_vitals)

    # Create subplots: vitals + prediction panel
    fig = make_subplots(
        rows=n_vital_rows + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[1] * n_vital_rows + [1.5],
        subplot_titles=[vital_configs[v]["name"] for v in available_vitals]
        + ["Sepsis Risk Predictions"],
    )

    # Plot each vital sign
    vital_color = COLORS["primary"]
    for row_idx, vital_key in enumerate(available_vitals, start=1):
        config = vital_configs[vital_key]
        values = vitals[vital_key]

        # Add normal range background
        low, high = config["normal"]
        fig.add_hrect(
            y0=low,
            y1=high,
            fillcolor=COLORS["success"],
            opacity=0.1,
            line_width=0,
            row=row_idx,
            col=1,
        )

        # Add vital sign trace
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=values,
                mode="lines+markers",
                name=config["name"],
                line=dict(color=vital_color, width=1.5),
                marker=dict(size=4),
                showlegend=False,
                hovertemplate=(
                    f"<b>{config['name']}</b><br>"
                    "Time: %{x}<br>"
                    f"Value: %{{y:.1f}} {config['unit']}<br>"
                    "<extra></extra>"
                ),
            ),
            row=row_idx,
            col=1,
        )

        # Update y-axis label
        fig.update_yaxes(
            title_text=config["unit"],
            row=row_idx,
            col=1,
            gridcolor=COLORS["card_border"],
            linecolor=COLORS["card_border"],
            tickfont=dict(color=COLORS["text_secondary"], size=10),
            title_font=dict(color=COLORS["text_secondary"], size=10),
        )

    # Plot predictions
    prediction_row = n_vital_rows + 1
    for idx, (model_name, preds) in enumerate(predictions.items()):
        color = colorway[idx % len(colorway)]
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=preds,
                mode="lines",
                name=model_name,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.1])}",
                hovertemplate=(
                    f"<b>{model_name}</b><br>"
                    "Time: %{x}<br>"
                    "Risk: %{y:.1%}<br>"
                    "<extra></extra>"
                ),
            ),
            row=prediction_row,
            col=1,
        )

    # Add risk threshold line
    fig.add_hline(
        y=0.5,
        line=dict(color=COLORS["warning"], width=1, dash="dash"),
        annotation_text="Alert Threshold",
        annotation_position="right",
        annotation_font=dict(color=COLORS["warning"], size=10),
        row=prediction_row,
        col=1,
    )

    fig.update_yaxes(
        title_text="Risk Probability",
        range=[0, 1],
        row=prediction_row,
        col=1,
        gridcolor=COLORS["card_border"],
        linecolor=COLORS["card_border"],
        tickfont=dict(color=COLORS["text_secondary"]),
        title_font=dict(color=COLORS["text_secondary"]),
    )

    # Update x-axis for bottom subplot
    fig.update_xaxes(
        title_text="Time",
        row=prediction_row,
        col=1,
        gridcolor=COLORS["card_border"],
        linecolor=COLORS["card_border"],
        tickfont=dict(color=COLORS["text_secondary"]),
        title_font=dict(color=COLORS["text_primary"]),
    )

    # Apply overall layout
    fig.update_layout(
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=COLORS["text_primary"],
        ),
        title=dict(
            text="Patient Vital Signs & Sepsis Risk Timeline",
            font=dict(size=16, color=COLORS["text_primary"]),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=COLORS["text_primary"]),
            bordercolor=COLORS["card_border"],
            x=1.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
        ),
        hovermode="x unified",
        height=150 * (n_vital_rows + 2),
        margin=dict(l=60, r=150, t=60, b=40),
    )

    # Style subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.color = COLORS["text_primary"]
        annotation.font.size = 11

    return fig
