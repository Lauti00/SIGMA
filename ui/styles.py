
import math
from urllib.parse import quote

import streamlit as st


def _wave_data_uri():
    """Onda ilustrativa de SIGMA: señal ruidosa y señal filtrada."""
    xs = range(0, 561, 4)

    def path(fn):
        return " ".join(
            f"{'M' if i == 0 else 'L'}{x},{fn(x):.1f}"
            for i, x in enumerate(xs)
        )

    clean = lambda x: 80 + 34 * math.sin(x / 22)
    noisy = lambda x: (
        clean(x)
        + 13 * math.sin(x / 2.7)
        + 5 * math.sin(x / 1.3)
    )

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 560 160" fill="none">'
        f'<path d="{path(noisy)}" stroke="#4F6BED" '
        'stroke-opacity="0.65" stroke-width="1.5"/>'
        f'<path d="{path(clean)}" stroke="#F08A24" '
        'stroke-width="3" stroke-linecap="round"/>'
        "</svg>"
    )

    return "data:image/svg+xml;utf8," + quote(svg)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --sigma-blue: #4F6BED;
    --sigma-orange: #F08A24;
    --sigma-teal: #18A999;
    --sigma-bg: #0B0F19;
    --sigma-panel: #121826;
    --sigma-panel-soft: #171F30;
    --sigma-line: rgba(255,255,255,0.09);
    --sigma-muted: #9AA6BA;
}

.stApp,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp p,
.stApp label,
.stApp li,
.stApp button,
.stApp input,
.stApp textarea {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.8rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #0B0F19;
    border-right: 1px solid var(--sigma-line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* HERO PRINCIPAL */

.sigma-hero {
    position: relative;
    overflow: hidden;
    border-radius: 22px;
    padding: 38px 42px;
    margin-bottom: 1.5rem;
    border: 1px solid var(--sigma-line);
    background:
        radial-gradient(900px 320px at 88% -30%, rgba(79,107,237,0.38), transparent 70%),
        radial-gradient(700px 260px at -5% 130%, rgba(240,138,36,0.20), transparent 70%),
        #111726;
}

.sigma-hero::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 56%;
    background: url("__WAVE__") no-repeat right center / contain;
    pointer-events: none;
    -webkit-mask-image: linear-gradient(to right, transparent, #000 45%);
    mask-image: linear-gradient(to right, transparent, #000 45%);
}

.sigma-hero .inner {
    position: relative;
    z-index: 1;
    max-width: 620px;
}

.sigma-badge {
    display: inline-block;
    padding: 5px 12px;
    margin-bottom: 14px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #B9C4FF;
    background: rgba(79,107,237,0.18);
    border: 1px solid rgba(79,107,237,0.45);
}

.sigma-hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1.4px;
    line-height: 1.05;
    margin: 0 0 12px 0;
    padding: 0;
    background: linear-gradient(90deg, #fff 30%, #9FB0FF);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sigma-hero h1 span {
    font-weight: 500;
}

.sigma-hero p {
    font-size: 1.02rem;
    color: #C1CAD9;
    margin: 0;
    line-height: 1.65;
}

.sigma-hero.compact {
    padding: 25px 32px;
}

.sigma-hero.compact h1 {
    font-size: 2rem;
    margin-bottom: 5px;
}

.sigma-hero.compact p {
    font-size: 0.92rem;
}

.sigma-hero.compact .sigma-badge {
    display: none;
}

.sigma-hero.compact::after {
    width: 40%;
    opacity: 0.7;
}

/* MARCA Y SIDEBAR */

.sigma-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 18px 0;
}

.sigma-logo {
    width: 44px;
    height: 44px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.55rem;
    font-weight: 800;
    color: #fff;
    background: linear-gradient(135deg, #4F6BED, #7B5CF0 60%, #F08A24);
    box-shadow: 0 6px 20px rgba(79,107,237,0.35);
}

.sigma-brand b {
    display: block;
    font-size: 1.15rem;
    letter-spacing: 0.08em;
    line-height: 1.1;
}

.sigma-brand small {
    color: var(--sigma-muted);
    font-size: 0.72rem;
}

.sigma-side-label {
    margin: 8px 0 5px 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem;
    font-weight: 700;
    color: #8FA0FF;
}

/* SECCIONES */

.sigma-section {
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 1.7rem 0 0.45rem 0;
}

.sigma-section .bar {
    width: 4px;
    height: 23px;
    flex-shrink: 0;
    border-radius: 3px;
    background: linear-gradient(#4F6BED, #18A999);
}

.sigma-section h3 {
    margin: 0;
    padding: 0;
    font-size: 1.18rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}

.sigma-caption {
    color: var(--sigma-muted);
    font-size: 0.84rem;
    margin: 0 0 0.9rem 15px;
}

/* TARJETAS */

.sigma-card {
    background: var(--sigma-panel);
    border: 1px solid var(--sigma-line);
    border-radius: 15px;
    padding: 19px;
    height: 100%;
}

.sigma-card h4 {
    margin: 0 0 7px 0;
    font-size: 0.98rem;
    font-weight: 700;
}

.sigma-card p {
    margin: 0;
    color: var(--sigma-muted);
    font-size: 0.86rem;
    line-height: 1.55;
}

.sigma-card-accent {
    border-top: 2px solid var(--sigma-blue);
}

.sigma-card-orange {
    border-top: 2px solid var(--sigma-orange);
}

.sigma-card-teal {
    border-top: 2px solid var(--sigma-teal);
}

/* DASHBOARD */

.sigma-dashboard {
    background:
        radial-gradient(600px 180px at 100% 0%, rgba(79,107,237,0.12), transparent 70%),
        #111827;
    border: 1px solid var(--sigma-line);
    border-radius: 18px;
    padding: 22px 24px;
    margin: 0.5rem 0 1.1rem 0;
}

.sigma-dashboard-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 15px;
}

.sigma-dashboard-heading h3 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
}

.sigma-dashboard-heading p {
    margin: 4px 0 0 0;
    color: var(--sigma-muted);
    font-size: 0.8rem;
}

.sigma-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    white-space: nowrap;
}

.sigma-status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
}

.sigma-status-ready {
    color: #62D6B5;
    background: rgba(24,169,153,0.13);
    border: 1px solid rgba(24,169,153,0.28);
}

.sigma-status-pending {
    color: #F5B96C;
    background: rgba(240,138,36,0.12);
    border: 1px solid rgba(240,138,36,0.25);
}

.sigma-status-idle {
    color: #AAB5C8;
    background: rgba(148,163,184,0.10);
    border: 1px solid var(--sigma-line);
}

.sigma-dashboard-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
}

.sigma-dashboard-item {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 13px 15px;
    min-width: 0;
}

.sigma-dashboard-item span {
    display: block;
    color: var(--sigma-muted);
    font-size: 0.72rem;
    margin-bottom: 7px;
}

.sigma-dashboard-item strong {
    display: block;
    color: #F3F6FC;
    font-size: 1.12rem;
    font-weight: 700;
    overflow-wrap: anywhere;
}

.sigma-dashboard-item small {
    display: block;
    color: #8491A6;
    font-size: 0.7rem;
    margin-top: 4px;
}

.sigma-dashboard-note {
    color: var(--sigma-muted);
    font-size: 0.8rem;
    margin-top: 13px;
}

/* CHIPS Y PIE */

.sigma-chip {
    display: inline-block;
    padding: 5px 11px;
    margin: 0 6px 6px 0;
    border-radius: 999px;
    font-size: 0.8rem;
    color: #C5CEFF;
    border: 1px solid rgba(79,107,237,0.4);
    background: rgba(79,107,237,0.12);
}

.sigma-footer {
    margin-top: 3rem;
    padding-top: 1rem;
    text-align: center;
    color: #8491A6;
    opacity: 0.65;
    font-size: 0.76rem;
    border-top: 1px solid var(--sigma-line);
}

/* COMPONENTES STREAMLIT */

[data-testid="stMetric"] {
    background: var(--sigma-panel);
    border: 1px solid var(--sigma-line);
    border-top: 2px solid var(--sigma-blue);
    border-radius: 13px;
    padding: 14px 17px;
    min-height: 105px;
}

[data-testid="stMetricLabel"] {
    color: var(--sigma-muted);
    font-size: 0.78rem;
}

[data-testid="stMetricValue"] {
    font-weight: 700;
    font-size: 1.35rem;
}

[data-testid="stPlotlyChart"] {
    background: var(--sigma-panel);
    border: 1px solid var(--sigma-line);
    border-radius: 14px;
    padding: 8px 8px 2px 8px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    padding: 6px;
    border-radius: 12px;
    background: var(--sigma-panel);
    border: 1px solid var(--sigma-line);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding: 0 18px;
    border-radius: 8px;
    background: transparent;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: rgba(79,107,237,0.25);
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

button[kind="primary"] {
    border: 0;
    border-radius: 10px;
    font-weight: 600;
    background: linear-gradient(135deg, #4F6BED, #7B5CF0);
}

button[kind="primary"]:hover {
    box-shadow: 0 6px 18px rgba(79,107,237,0.4);
}

[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid var(--sigma-line);
}

[data-testid="stFileUploader"] {
    border-radius: 12px;
}

hr {
    border-color: var(--sigma-line);
}

@media (max-width: 900px) {
    .sigma-dashboard-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .sigma-hero {
        padding: 28px 25px;
    }

    .sigma-hero h1 {
        font-size: 2.3rem;
    }
}

@media (max-width: 560px) {
    .sigma-dashboard-grid {
        grid-template-columns: 1fr 1fr;
    }

    .sigma-dashboard {
        padding: 16px;
    }

    .sigma-hero::after {
        opacity: 0.25;
        width: 75%;
    }
}
</style>
""".replace("__WAVE__", _wave_data_uri())


def apply_styles():
    """Inyecta el CSS global de SIGMA."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, badge="Procesamiento Digital de Señales", compact=False):
    """Banner principal de SIGMA."""
    brand_word, _, rest = title.partition(" ")
    rest_html = f" <span>{rest}</span>" if rest else ""
    css_class = "sigma-hero compact" if compact else "sigma-hero"

    st.markdown(
        f'<div class="{css_class}"><div class="inner">'
        f'<div class="sigma-badge">{badge}</div>'
        f"<h1>{brand_word}{rest_html}</h1>"
        f"<p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def brand():
    """Logo y nombre para el panel lateral."""
    st.markdown(
        '<div class="sigma-brand">'
        '<div class="sigma-logo">Σ</div>'
        '<div><b>SIGMA</b><small>Audio Lab · PDS</small></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def side_label(text):
    st.markdown(
        f'<div class="sigma-side-label">{text}</div>',
        unsafe_allow_html=True,
    )


def section(title, caption=None):
    """Título de sección con barra de acento y bajada opcional."""
    html = (
        '<div class="sigma-section">'
        '<div class="bar"></div>'
        f'<h3>{title}</h3>'
        '</div>'
    )

    if caption:
        html += f'<div class="sigma-caption">{caption}</div>'

    st.markdown(html, unsafe_allow_html=True)


def chips(items):
    st.markdown(
        "".join(
            f'<span class="sigma-chip">{item}</span>'
            for item in items
        ),
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        '<div class="sigma-footer">'
        'SIGMA · Audio Lab — Procesamiento Digital de Señales'
        '</div>',
        unsafe_allow_html=True,
    )
