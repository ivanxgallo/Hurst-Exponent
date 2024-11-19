import numpy as np
from scipy.optimize import curve_fit


def mfdfa(ts, q=2, min_scale = 4, max_scale=1000, order=1, test=False, last_point=100):
    # Asegurarse de que la serie sea un array de NumPy
    ts = np.asarray(ts)
    N = len(ts)

    # Calcular el perfil acumulado (integración de la serie restando la media)
    mean_ts = np.mean(ts)
    profile = np.cumsum(ts - mean_ts)

    # Definir las escalas (longitudes de segmento)
    scales = np.arange(min_scale, max_scale)
    flucts = []

    for s in scales:
        # Dividir el perfil en segmentos de longitud s
        num_segments = N // s
        F_nu = []

        for v in range(0, num_segments):
            # Obtener el segmento
            segment = profile[v * s:(v + 1) * s]
            # Ajustar un polinomio de orden 'order' al segmento y restar la tendencia
            x = np.arange(s)
            poly_coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(poly_coeffs, x)
            detrended_segment = segment - trend
            # Calcular la varianza de las diferencias
            F_nu.append(np.mean(detrended_segment**2))

        '''
        # Analizar los segmentos de atrás hacia adelante
        for v in range(num_segments):
            # Obtener el segmento desde el final
            segment = profile[N - (v + 1) * s : N - v * s]
            # Ajustar un polinomio de orden 'order' al segmento y restar la tendencia
            x = np.arange(s)
            poly_coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(poly_coeffs, x)
            detrended_segment = segment - trend
            # Calcular la varianza de las diferencias
            F_nu.append(np.mean(detrended_segment**2))

        '''

        # Calcular la función de fluctuación en la escala s
        if q != 0:
            F_s = (np.mean(np.array(F_nu) ** (q / 2))) ** (1 / q)
        else:
            # Caso especial para q=0
            F_s = np.exp(0.5 * np.mean(np.log(F_nu)))
        flucts.append(F_s)

    # Ajustar una recta en la escala log-log para encontrar el exponente h(q)
    """
    log_scales = np.log(scales)
    log_flucts = np.log(flucts)
    coeffs, cov = np.polyfit(log_scales, log_flucts, 1, cov=True)
    h_q = coeffs[0]  # El exponente de Hurst generalizado
    h_q_error = np.sqrt(cov[0][0])  # Error estándar de la pendiente
    """
    _, __, params = XY_fit(scales[:last_point], flucts[:last_point])
    if test is False:
        return params[-1]
    else:
        return scales, flucts

def pl(x, A, a):
    return A*x**a

def XY_fit(x_data, y_data, func=pl, x_range=[10, 1000], n_points=1000):
    # Fit using curve_fit
    params, _ = curve_fit(func, x_data, y_data)

    # Generate x values in the specified range
    x_fit = np.linspace(x_range[0], x_range[1], n_points)
    # Calculate y values of the fitted function with obtained parameters
    y_fit = func(x_fit, *params)

    return x_fit, y_fit, params


def count1bin(n):
    # Convierte el número a binario, cuenta los '1' y retorna el resultado
    return bin(n).count('1')

def bimus(a=0.75, n_max=13):
    N = 2**n_max
    t = range(1, N+1)
    ts = [a**count1bin(k)*(1 - a)**(n_max - count1bin(k)) for k in t]
    return t, ts

def h_bimus(a=0.75, q=2.):
    return (1 - np.log(a**q + (1-a)**q)/(np.log(2)))/q


def smooth_growing_window(F_s):
    """
    Suaviza un array F_s mediante un promedio móvil con ventanas crecientes.

    Parámetros:
    - F_s (array): Array de valores a suavizar.

    Retorna:
    - smooth_F_s (array): Array suavizado.
    """
    N = len(F_s)
    smooth_F_s = np.zeros(N)

    for i in range(N):
        # Definir el tamaño de la ventana creciente
        window_size = max(1, int(np.sqrt(i + 1)))  # Usa raíz cuadrada de la posición para ventana creciente
        start = max(0, i - window_size)
        end = i + 1
        # Calcular el promedio en la ventana actual
        smooth_F_s[i] = np.mean(F_s[start:end])

    return smooth_F_s


def smooth_symmetric_window(F_s):
    """
    Suaviza un array F_s mediante un promedio móvil simétrico con ventanas crecientes
    alrededor de cada punto, ajustando el tamaño en los extremos del array.

    Parámetros:
    - F_s (array): Array de valores a suavizar.

    Retorna:
    - smooth_F_s (array): Array suavizado.
    """
    N = len(F_s)
    smooth_F_s = np.zeros(N)

    for i in range(N):
        # Definir el tamaño de la ventana creciente
        window_size_s = max(1, int((i + 1)))  # Usa raíz cuadrada para ventana creciente
        window_size_e = max(1, int((i + 1)))
        # Determinar los límites de la ventana simétrica alrededor de `i`
        start = max(0, i - window_size_s)
        end = min(N, i + window_size_e + 1)  # `+1` para incluir el punto `end`

        # Calcular el promedio en la ventana actual
        smooth_F_s[i] = np.mean(F_s[start:end])

    return smooth_F_s


def equalize_scale_density(F_s, scales, points_per_interval=10):
    """
    Reescala el array F_s para que tenga la misma cantidad de elementos
    en intervalos logarítmicos definidos en el array de escalas.

    Parámetros:
    - F_s (array): Array de valores de fluctuación.
    - scales (array): Array de escalas correspondientes a cada valor en F_s.
    - points_per_interval (int): Número de puntos a seleccionar en cada intervalo logarítmico.

    Retorna:
    - equalized_scales (array): Escalas con densidad uniforme.
    - equalized_F_s (array): Array F_s reescalado.
    """
    # Definir los límites logarítmicos de los intervalos
    log_min = np.log10(scales[0])
    log_max = np.log10(scales[-1])
    num_intervals = int((log_max - log_min) + 1)  # Número de intervalos en escala log10

    # Crear listas para almacenar los resultados
    equalized_scales = []
    equalized_F_s = []

    for i in range(num_intervals):
        # Definir el rango logarítmico para el intervalo actual
        interval_min = 10 ** (log_min + i)
        interval_max = 10 ** (log_min + i + 1)

        # Seleccionar los puntos en el rango actual
        indices_in_range = np.where((scales >= interval_min) & (scales < interval_max))[0]

        # Asegurarse de que el rango tenga suficientes puntos para aplicar el promedio
        if len(indices_in_range) >= 2:
            # Selección uniforme de puntos en el intervalo actual
            selected_indices = np.linspace(indices_in_range[0], indices_in_range[-1], points_per_interval, dtype=int)
            equalized_scales.extend(scales[selected_indices].tolist())
            equalized_F_s.extend(F_s[selected_indices].tolist())

    return np.array(equalized_scales), np.array(equalized_F_s)


def generate_noise_series(n_points, beta):
    """
    Generates a time series of noise with a frequency spectrum characterized by an exponent beta.

    Parameters:
    - n_points: int, number of points in the time series.
    - beta: float, exponent of the frequency spectrum.

    Returns:
    - series: array of floats, generated time series.
    """
    # Create Fourier frequencies and apply the desired power spectrum exponent
    freqs = np.fft.fftfreq(n_points)
    # Generate random phase and magnitude proportional to the desired power
    f_amplitude = np.abs(freqs) ** (beta / 2.0)
    f_amplitude[0] = 0  # Set frequency 0 to zero to avoid infinities
    # Create the signal in the Fourier domain
    noise_f = f_amplitude * np.exp(2j * np.pi * np.random.rand(n_points))
    # Transform back to the time domain and take only the real part
    series = np.fft.ifft(noise_f).real
    return series


def compute_spectrum(ts):
    """
    Calcula el espectro de frecuencia de una serie de tiempo.

    Parámetros:
    - ts: array de floats, la serie de tiempo para la cual se calculará el espectro.

    Retorna:
    - f: array de floats, las frecuencias correspondientes.
    - spectra: array de floats, el espectro de potencia para cada frecuencia.
    """
    # Número de puntos en la serie de tiempo
    n_points = len(ts)
    # Transformada de Fourier de la serie de tiempo
    fft_result = np.fft.fft(ts)
    # Frecuencias asociadas
    f = np.fft.fftfreq(n_points)
    # Cálculo del espectro de potencia (módulo al cuadrado de la transformada)
    spectrum = np.abs(fft_result) ** 2
    # Retornar frecuencias y espectro de potencia
    return f, spectrum
