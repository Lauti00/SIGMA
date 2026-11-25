from pathlib import Path

import streamlit as st

from ui.styles import chips, footer, section

DEMO_PATH = (
    Path(__file__).resolve().parent.parent
    / "samples"
    / "audio_examples"
    / "demo_tonos_440_2500Hz.wav"
)
DEMO_FLAG = "sigma_use_demo"

STEPS = (
    ("1", "Cargá tu audio", "Subí un WAV mono o estéreo desde el panel lateral."),
    ("2", "Analizá", "Mirá la forma de onda, el espectro FFT y los picos dominantes."),
    ("3", "Filtrá y compará", "Diseñá un FIR o IIR, escuchá el resultado y descargalo."),
)
FEATURES = (
    "FIR · Kaiser",
    "IIR · Butterworth",
    "FFT con ventanas",
    "Ruido con SNR controlada",
    "MSE y SNR con referencia limpia",
    "Gráficos interactivos",
)


class InMemoryFile:
    """Imita lo mínimo de un archivo subido: el resto del código solo usa getvalue()."""

    def __init__(self, data, name):
        self._data = data
        self.name = name

    def getvalue(self):
        return self._data


def load_demo_file():
    """Devuelve el WAV de ejemplo, o None si el archivo no existe."""
    if not DEMO_PATH.is_file():
        return None
    return InMemoryFile(DEMO_PATH.read_bytes(), DEMO_PATH.name)


def _start_demo():
    st.session_state[DEMO_FLAG] = True


def _stop_demo():
    st.session_state[DEMO_FLAG] = False


def select_active_file(uploaded_file):
    """Decide qué audio se usa. Llamar dentro de `with st.sidebar`.

    Prioridad: archivo subido > audio de ejemplo > ninguno.
    """
    if uploaded_file is not None:
        st.session_state[DEMO_FLAG] = False
        return uploaded_file

    if st.session_state.get(DEMO_FLAG):
        demo = load_demo_file()
        if demo is not None:
            st.info("Usando el audio de ejemplo (440 Hz + 2500 Hz).")
            st.button("Salir del ejemplo", on_click=_stop_demo, width="stretch")
            return demo
        st.session_state[DEMO_FLAG] = False

    return None


def render_welcome():
    """Pantalla inicial cuando todavía no hay audio cargado."""
    cta_button, cta_text = st.columns([1, 2], vertical_alignment="center")
    if DEMO_PATH.is_file():
        cta_button.button("▶ Probar con audio de ejemplo", type="primary", on_click=_start_demo, width="stretch")
        cta_text.caption(
            "Dos tonos (440 Hz y 2500 Hz) a 16 kHz durante 3 s. "
            "Ideal para ver cómo un pasabajo conserva uno y atenúa el otro. También podés subir tu propio WAV desde el panel lateral."
        )
    else:
        st.warning("No se encontró el audio de ejemplo en samples/audio_examples/.")

    section("Cómo funciona", "Tres pasos, del archivo al resultado")
    columns = st.columns(3)
    for column, (number, title, text) in zip(columns, STEPS):
        column.markdown(
            f'<div class="sigma-card"><div class="n">{number}</div><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

    section("Qué incluye")
    chips(FEATURES)
    footer()