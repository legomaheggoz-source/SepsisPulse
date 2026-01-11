"""Aurora Solar-inspired light theme for SepsisPulse dashboard."""

import streamlit as st


# Aurora Solar-inspired light color palette
COLORS = {
    "background": "#f8fafb",
    "card_bg": "#ffffff",
    "card_border": "#e0e4e8",
    "primary": "#0966d2",
    "secondary": "#6e7681",
    "success": "#1a7f37",
    "warning": "#b08500",
    "danger": "#da3633",
    "text_primary": "#24292f",
    "text_secondary": "#57606a",
}


def get_plotly_template() -> dict:
    """
    Get a Plotly template matching the Aurora light theme.

    Returns:
        dict: Plotly template configuration dictionary.
    """
    return {
        "layout": {
            "paper_bgcolor": COLORS["card_bg"],
            "plot_bgcolor": COLORS["background"],
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
                "bgcolor": "rgba(255,255,255,0.9)",
                "font": {"color": COLORS["text_primary"]},
                "bordercolor": COLORS["card_border"],
            },
            "colorway": [
                COLORS["primary"],
                COLORS["success"],
                COLORS["warning"],
                COLORS["danger"],
                "#8250df",
                "#bf3989",
                "#0550ae",
                "#116329",
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
    Apply Aurora light theme CSS to Streamlit application.

    Injects custom CSS styles that override Streamlit's default styling
    to match the Aurora Solar-inspired light theme.
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

        h1 {{
            color: {COLORS["primary"]} !important;
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
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }}

        .stMetric label {{
            color: {COLORS["text_secondary"]} !important;
        }}

        .stMetric [data-testid="stMetricValue"] {{
            color: {COLORS["primary"]} !important;
        }}

        .stMetric [data-testid="stMetricDelta"] svg {{
            stroke: {COLORS["success"]};
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {COLORS["primary"]};
            color: white;
            border: none;
            border-radius: 6px;
            transition: background-color 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: #0550ae;
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
            background-color: {COLORS["background"]};
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
