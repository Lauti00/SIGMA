import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1300px; }
[data-testid="stMetric"] {
    background: #161B26;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { opacity: 0.7; }
.sigma-hero h1 { font-size: 2.2rem; margin-bottom: 0; letter-spacing: -0.5px; }
.sigma-hero p  { opacity: 0.7; margin-top: 4px; }
button[kind="primary"] { border-radius: 10px; font-weight: 600; }
</style>
"""


def apply_styles():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(
        f'<div class="sigma-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )