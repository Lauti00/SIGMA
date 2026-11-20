# Guía de experimentos y defensa — SIGMA

## 1. Pregunta del proyecto

¿Cómo cambia el contenido temporal y frecuencial de una señal WAV al aplicarle un filtro digital configurable? SIGMA permite observar la señal antes y después, escuchar el resultado y contrastarlo con la respuesta en frecuencia del filtro.

## 2. Experimento controlado: dos tonos

Usen `samples/audio_examples/demo_tonos_440_2500Hz.wav`. Es una suma de tonos de 440 Hz y 2500 Hz, muestreada a 16 000 Hz durante 3 segundos.

1. Carguen el WAV y comprueben que informa Fs = 16 000 Hz, 48 000 muestras y 3 s.
2. Seleccionen ventana rectangular y verifiquen picos cerca de 440 Hz y 2500 Hz.
3. Apliquen un pasabajo FIR con corte de 1000 Hz.
4. Comparen el espectro: debe permanecer el componente de 440 Hz y bajar el de 2500 Hz.
5. Repitan con Butterworth IIR y comparen la transición, respuesta y audio.

La frecuencia de Nyquist es Fs/2 = 8000 Hz. La separación entre bins de FFT es Δf = Fs/N; en este archivo, 16 000/48 000 = 1/3 Hz.

## 3. Experimento controlado: contaminación conocida

1. Carguen una voz limpia en WAV mono o estéreo.
2. En el panel lateral abran **Experimento: agregar ruido** y activen **Generar contaminación sintética**.
3. Elijan **Tono sinusoidal**, indiquen 50 Hz y seleccionen una SNR objetivo, por ejemplo 10 dB.
4. Observen que la aplicación conserva una copia limpia como referencia y crea una mezcla contaminada reproducible. Una SNR menor significa más ruido en relación con la señal útil.
5. Configuren un filtro IIR Butterworth rechaza banda, por ejemplo entre 49 y 51 Hz, y comparen la referencia limpia, la señal contaminada y la procesada en el tiempo, el espectro y la escucha.
6. Repitan con ruido blanco. Un filtro pasabajo puede reducir parte del ruido, pero también elimina componentes agudas legítimas de la voz.

El ruido blanco usa una semilla reproducible asociada al archivo; los tonos también se generan de forma determinista. La suma se escala en conjunto si supera el rango de amplitud permitido; esa ganancia común mantiene la SNR. Este limitador evita clipping digital, pero reduce el nivel general del audio.

## 4. Qué significan los gráficos

- **Dominio temporal:** cada punto representa una muestra. El eje X es `n/Fs`, en segundos; el eje Y es amplitud normalizada.
- **Dominio frecuencial:** la FFT muestra amplitud por bin. Si una componente no cae exactamente en un bin, la energía se distribuye entre bins vecinos; eso se llama leakage espectral.
- **Ventanas:** Hann y Hamming reducen lóbulos laterales y el leakage lejos del tono, pero ensanchan el pico principal. La amplitud se divide por la suma de los coeficientes de la ventana para conservar la escala de los tonos.
- **Respuesta del filtro:** se expresa en dB, con 0 dB como ganancia unitaria. La zona de transición no es un corte vertical ideal.

## 5. FIR e IIR

Un FIR se describe mediante la convolución `y[n] = Σ b[k] x[n-k]`. En SIGMA, NumPy calcula la convolución directa para señales de hasta 200 000 muestras; para archivos largos se usa la convolución por FFT de `scipy.signal.fftconvolve`, que obtiene el mismo resultado de forma más eficiente. Los coeficientes los diseña `scipy.signal.firwin` con ventana Kaiser. Para facilitar comparar los sonidos, el resultado se recorta centrado y se compensa el retardo de grupo del FIR lineal en fase; es procesamiento fuera de línea, no filtrado en tiempo real.

Un IIR contiene muestras anteriores de entrada y de salida: `y[n] = (Σ b[k]x[n-k] - Σ a[k]y[n-k])/a[0]`. SciPy diseña los coeficientes Butterworth en secciones de segundo orden (SOS); SIGMA aplica esas secciones muestra a muestra con la forma directa II transpuesta en señales de hasta 200 000 muestras. Para las más largas, `scipy.signal.sosfilt` realiza la misma recurrencia en código optimizado. Cada sección comienza en un estado estacionario coherente con la primera muestra. El procesamiento es causal.

Butterworth se usa como primera alternativa porque su magnitud en la banda pasante es suave, sin ondulaciones intencionales. FIR e IIR pueden tener distinto orden y distinta transición, latencia y respuesta de fase; no se deben comparar solo por el valor numérico del orden.

## 6. Métricas (solo con referencia limpia)

La aplicación acepta opcionalmente otro WAV como referencia limpia. Para que la comparación tenga sentido, la referencia debe corresponder a la señal objetivo, estar alineada muestra a muestra, tener igual Fs y duración, y usar la misma escala. SIGMA comprueba Fs y cantidad de muestras; la alineación y la limpieza deben verificarlas quienes realizan el experimento.

`MSE = mean((referencia - estimación)^2)` mide error cuadrático promedio. `SNR = 10 log10(Preferencia / Perror)` compara potencia de señal con potencia de error. No usen el audio ruidoso original como referencia limpia: el filtro cambiaría contenido legítimo y las métricas medirían ese cambio junto con el ruido.

## 7. Qué validar y qué declarar

- La detección de tonos se valida con frecuencias conocidas en el WAV sintético.
- La atenuación se valida tanto en la FFT de salida como en la respuesta en frecuencia.
- FIR: diseño Kaiser asistido por SciPy; convolución realizada en el código con NumPy.
- IIR: diseño Butterworth asistido por SciPy; ecuaciones en diferencias implementadas en el código por SOS.
- WAV y FFT: lectura/escritura apoyada por SciPy; FFT y ejes de muestra/frecuencia calculados con NumPy.
- Los filtros son herramientas de procesamiento, no eliminan automáticamente ruido sin afectar otras componentes. El resultado auditivo depende del contenido y los parámetros.
