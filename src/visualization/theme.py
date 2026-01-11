"""Aurora Solar-inspired dark theme for SepsisPulse dashboard."""

import streamlit as st


# Aurora Solar-inspired color palette
COLORS = {
    "background": "#0d1117",
    "card_bg": "#161b22",
    "card_border": "#30363d",
    "primary": "#58a6ff",
    "secondary": "#8b949e",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f85149",
    "text_primary": "#f0f6fc",
    "text_secondary": "#8b949e",
}


def get_plotly_template() -> dict:
    """
    Get a Plotly template matching the Aurora dark theme.

    Returns:
        dict: Plotly template configuration dictionary.
    """
    return {
        "layout": {
            "paper_bgcolor": COLORS["background"],
            "plot_bgcolor": COLORS["card_bg"],
            "font": {
                "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "color": COLORS["text_primary"],
                "size": 12,
            },
            "title": {
                "font": {
                    "size": 16,
                    "color": COLORS["text_primary"],
                },
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {
                "gridcolor": COLORS["card_border"],
                "linecolor": COLORS["card_border"],
                "tickcolor": COLORS["text_secondary"],
                "tickfont": {"color": COLORS["text_secondary"]},
                "title": {"font": {"color": COLORS["text_primary"]}},
                "zerolinecolor": COLORS["card_border"],
            },
            "yaxis": {
                "gridcolor": COLORS["card_border"],
                "linecolor": COLORS["card_border"],
                "tickcolor": COLORS["text_secondary"],
                "tickfont": {"color": COLORS["text_secondary"]},
                "title": {"font": {"color": COLORS["text_primary"]}},
                "zerolinecolor": COLORS["card_border"],
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "font": {"color": COLORS["text_primary"]},
                "bordercolor": COLORS["card_border"],
            },
            "colorway": [
                COLORS["primary"],
                COLORS["success"],
                COLORS["warning"],
                COLORS["danger"],
                "#a371f7",
                "#f778ba",
                "#79c0ff",
                "#7ee787",
            ],
            "hoverlabel": {
                "bgcolor": COLORS["card_bg"],
                "bordercolor": COLORS["card_border"],
                "font": {"color": COLORS["text_primary"]},
            },
            "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
        },
    }


def apply_aurora_theme() -> None:
    """
    Apply Aurora dark theme CSS to Streamlit application.

    Injects custom CSS styles that override Streamlit's default styling
    to match the Aurora Solar-inspired dark theme.
    """
    css = f"""
    <style>
        /* Main app background */
        .stApp {{
            background-color: {COLORS["background"]};
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: {COLORS["card_bg"]};
            border-right: 1px solid {COLORS["card_border"]};
        }}

        section[data-testid="stSidebar"] .stMarkdown {{
            color: {COLORS["text_primary"]};
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS["text_primary"]} !important;
        }}

        /* Text elements */
        p, span, label, .stMarkdown {{
            color: {COLORS["text_secondary"]};
        }}

        /* Cards and containers */
        .stMetric {{
            background-color: {COLORS["card_bg"]};
            border: 1px solid {COLORS["card_border"]};
            border-radius: 8px;
            padding: 16px;
        }}

        .stMetric label {{
            color: {COLORS["text_secondary"]} !important;
        }}

        .stMetric [data-testid="stMetricValue"] {{
            color: {COLORS["text_primary"]} !important;
        }}

        .stMetric [data-testid="stMetricDelta"] svg {{
            stroke: {COLORS["success"]};
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {COLORS["primary"]};
            color: {COLORS["text_primary"]};
            border: none;
            border-radius: 6px;
            transition: background-color 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: #4c9aed;
        }}

        /* Input fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {{
            background-color: {COLORS["card_bg"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["card_border"]};
            border-radius: 6px;
        }}

        /* Dataframe styling */
        .stDataFrame {{
            border: 1px solid {COLORS["card_border"]};
            border-radius: 8px;
            overflow: hidden;
        }}

        .stDataFrame [data-testid="stDataFrameResizable"] {{
            background-color: {COLORS["card_bg"]};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {COLORS["card_bg"]};
            border-bottom: 1px solid {COLORS["card_border"]};
        }}

        .stTabs [data-baseweb="tab"] {{
            color: {COLORS["text_secondary"]};
        }}

        .stTabs [aria-selected="true"] {{
            color: {COLORS["primary"]} !important;
            border-bottom: 2px solid {COLORS["primary"]} !important;
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {COLORS["card_bg"]};
            border: 1px solid {COLORS["card_border"]};
            border-radius: 8px;
            color: {COLORS["text_primary"]};
        }}

        .streamlit-expanderContent {{
            background-color: {COLORS["card_bg"]};
            border: 1px solid {COLORS["card_border"]};
            border-top: none;
            border-radius: 0 0 8px 8px;
        }}

        /* Progress bar */
        .stProgress > div > div {{
            background-color: {COLORS["primary"]};
        }}

        /* Slider */
        .stSlider [data-baseweb="slider"] {{
            background-color: {COLORS["card_border"]};
        }}

        .stSlider [data-testid="stThumbValue"] {{
            color: {COLORS["text_primary"]};
        }}

        /* Alert/info boxes */
        .stAlert {{
            background-color: {COLORS["card_bg"]};
            border: 1px solid {COLORS["card_border"]};
            border-radius: 8px;
        }}

        /* Divider */
        hr {{
            border-color: {COLORS["card_border"]};
        }}

        /* Code blocks */
        .stCodeBlock {{
            background-color: {COLORS["card_bg"]};
            border: 1px solid {COLORS["card_border"]};
            border-radius: 8px;
        }}

        /* Scrollbar styling */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}

        ::-webkit-scrollbar-track {{
            background: {COLORS["background"]};
        }}

        ::-webkit-scrollbar-thumb {{
            background: {COLORS["card_border"]};
            border-radius: 4px;
        }}

        ::-webkit-scrollbar-thumb:hover {{
            background: {COLORS["secondary"]};
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
