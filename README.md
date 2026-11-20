# SIGMA

Aplicación local para analizar y procesar archivos WAV como proyecto de Procesamiento Digital de Señales.

## Inicio (Windows / PowerShell)

Desde esta carpeta (`SIGMA`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Para crear el entorno desde cero, instalá Python 3.11 o superior y ejecutá:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

## Funcionalidades

- Carga WAV mono o estéreo; el estéreo se promedia a mono.
- Datos básicos, reproductor y forma de onda.
- Espectro FFT de un lado, con ventana rectangular, Hann o Hamming, y picos predominantes.
- Filtros FIR Kaiser y filtros IIR Butterworth: pasabajo, pasaalto, pasabanda y rechaza banda.
- Generación reproducible de ruido blanco o tonos sinusoidales a una SNR configurada, con protección contra clipping.
- Gráficos interactivos con hover, zoom, desplazamiento, selector de rango y comparación original/procesada.
- Reproducción y descarga WAV.
- MSE y SNR opcionales cuando se carga una referencia limpia alineada.

## Fundamento y responsabilidades de las bibliotecas

La interfaz presenta los gráficos con Plotly dentro de Streamlit. Se puede explorar cada curva con hover, zoom y desplazamiento; en la forma temporal hay además un selector de rango. Los parámetros del filtro actualizan los gráficos automáticamente en señales cortas, con opción de aplicar manualmente en archivos largos. El ruido artificial se escala según su potencia RMS para obtener la SNR seleccionada; la mezcla y la referencia reciben la misma ganancia para evitar clipping sin alterar esa SNR.

La lectura WAV y el diseño de filtros se apoyan en SciPy. FIR usa `np.convolve` para señales de hasta 200 000 muestras y `scipy.signal.fftconvolve` en audios más largos; ambos calculan la convolución discreta. IIR se desarrolla muestra a muestra con la forma directa II transpuesta para señales de hasta 200 000 muestras y usa `scipy.signal.sosfilt` para las más largas. Los coeficientes SOS se diseñan con `scipy.signal.butter`. La FFT usa `numpy.fft.rfft` y `rfftfreq` sobre un tramo elegido de hasta 30 s y 2 millones de muestras.

La FFT produce muestras discretas del espectro. Si una frecuencia no coincide con un bin, su energía se reparte en bins cercanos (leakage). La ventana Hann o Hamming atenúa los lóbulos laterales, aunque ensancha el pico. La amplitud se normaliza por la suma de la ventana y por eso un tono alineado conserva aproximadamente su amplitud pico. Los gráficos temporales de archivos largos muestran una envolvente de mínimos y máximos para evitar dibujar cada muestra en pantalla.

En el IIR, cada sección se inicializa en el estado estacionario que corresponde a una entrada constante igual a la primera muestra. El filtrado es causal y puede presentar transitorios cuando la señal cambia. El orden FIR impar solicitado se ajusta al siguiente orden par para mantener una cantidad impar de coeficientes y una respuesta de fase lineal simétrica.

Por defecto no se calcula SNR o MSE contra el audio original, porque no necesariamente es una referencia limpia. Se calculan al cargar un WAV limpio, alineado y con la misma frecuencia de muestreo y duración.

## Pruebas

Desde `SIGMA`, ejecutá:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Las pruebas cubren lectura WAV mono/estéreo, frecuencia y amplitud FFT, reducción de leakage, los ocho filtros, equivalencia del IIR manual frente a SciPy y métricas con referencia conocida.

## Audio de ejemplo

`samples/audio_examples/demo_tonos_440_2500Hz.wav` contiene dos tonos sintéticos (440 Hz y 2500 Hz) a 16 kHz. Usalo para observar ambos picos y luego comprobar que un pasabajo de 1000 Hz conserva el tono de 440 Hz y atenúa el de 2500 Hz.

Consultá [la guía de experimentos y defensa](docs/GUIA_EXPERIMENTOS.md) para el procedimiento, los fundamentos y los límites de interpretación.
