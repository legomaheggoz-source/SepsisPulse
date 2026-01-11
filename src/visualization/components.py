"""Streamlit UI components for SepsisPulse dashboard."""

from typing import Dict, Optional

import pandas as pd

from .theme import COLORS


def metric_card(title: str, value: str, delta: Optional[str] = None) -> str:
    """
    Create an HTML metric card component.

    Args:
        title: The metric title/label.
        value: The main metric value to display.
        delta: Optional delta/change indicator (e.g., "+5%" or "-2.3").

    Returns:
        str: HTML string for the metric card.
    """
    # Determine delta styling if provided
    delta_html = ""
    if delta is not None:
        # Determine if positive, negative, or neutral
        delta_stripped = delta.strip()
        if delta_stripped.startswith("+") or (
            delta_stripped[0].isdigit() and float(delta_stripped.rstrip("%")) > 0
        ):
            delta_color = COLORS["success"]
            delta_icon = "\u25b2"  # Up triangle
        elif delta_stripped.startswith("-"):
            delta_color = COLORS["danger"]
            delta_icon = "\u25bc"  # Down triangle
        else:
            delta_color = COLORS["secondary"]
            delta_icon = "\u25cf"  # Circle

        delta_html = f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 4px;
                font-size: 12px;
                color: {delta_color};
                margin-top: 4px;
            ">
                <span>{delta_icon}</span>
                <span>{delta}</span>
            </div>
        """

    return f"""
    <div style="
        background-color: {COLORS["card_bg"]};
        border: 1px solid {COLORS["card_border"]};
        border-radius: 8px;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 4px;
    ">
        <div style="
            font-size: 12px;
            font-weight: 500;
            color: {COLORS["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        ">
            {title}
        </div>
        <div style="
            font-size: 28px;
            font-weight: 600;
            color: {COLORS["text_primary"]};
            line-height: 1.2;
        ">
            {value}
        </div>
        {delta_html}
    </div>
    """


def alert_banner(message: str, severity: str) -> str:
    """
    Create an alert banner HTML component.

    Args:
        message: The alert message to display.
        severity: Alert severity level ('info', 'success', 'warning', 'danger').

    Returns:
        str: HTML string for the alert banner.
    """
    # Map severity to colors and icons
    severity_config = {
        "info": {
            "color": COLORS["primary"],
            "bg_color": f"{COLORS['primary']}1a",  # 10% opacity
            "icon": "\u2139",  # Info circle
        },
        "success": {
            "color": COLORS["success"],
            "bg_color": f"{COLORS['success']}1a",
            "icon": "\u2713",  # Check mark
        },
        "warning": {
            "color": COLORS["warning"],
            "bg_color": f"{COLORS['warning']}1a",
            "icon": "\u26a0",  # Warning triangle
        },
        "danger": {
            "color": COLORS["danger"],
            "bg_color": f"{COLORS['danger']}1a",
            "icon": "\u2716",  # X mark
        },
    }

    # Default to info if severity not recognized
    config = severity_config.get(severity.lower(), severity_config["info"])

    return f"""
    <div style="
        background-color: {config["bg_color"]};
        border: 1px solid {config["color"]};
        border-left: 4px solid {config["color"]};
        border-radius: 6px;
        padding: 12px 16px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        margin: 8px 0;
    ">
        <div style="
            font-size: 18px;
            color: {config["color"]};
            line-height: 1;
            flex-shrink: 0;
        ">
            {config["icon"]}
        </div>
        <div style="
            font-size: 14px;
            color: {COLORS["text_primary"]};
            line-height: 1.5;
            flex-grow: 1;
        ">
            {message}
        </div>
    </div>
    """


def model_comparison_table(metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a model comparison table from metrics dictionary.

    Args:
        metrics: Dictionary mapping model names to their metrics dictionaries.
                 Each inner dict should contain metric names as keys and
                 numeric values as values.
                 Example:
                 {
                     "XGBoost": {"AUROC": 0.89, "Sensitivity": 0.82, "Specificity": 0.91},
                     "LSTM": {"AUROC": 0.87, "Sensitivity": 0.79, "Specificity": 0.88}
                 }

    Returns:
        pd.DataFrame: Formatted comparison table with models as rows and metrics as columns.
    """
    if not metrics:
        return pd.DataFrame()

    # Convert nested dict to DataFrame
    df = pd.DataFrame(metrics).T
    df.index.name = "Model"

    # Reset index to make Model a column
    df = df.reset_index()

    # Format numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
    for col in numeric_cols:
        # Check if values appear to be percentages (0-1 range)
        if df[col].max() <= 1.0 and df[col].min() >= 0.0:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")
        else:
            df[col] = df[col].apply(lambda x: f"{x:.2f}")

    # Sort by first numeric metric column descending (usually the primary metric)
    if len(df.columns) > 1:
        first_metric_col = df.columns[1]
        df = df.sort_values(by=first_metric_col, ascending=False)

    return df
