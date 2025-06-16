import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def simur_getFFTpower(FFT, normalize=True):
    n = len(FFT)
    FFTpow = FFT * np.conjugate(FFT)
    FFTpow = FFTpow.real
    if normalize:
        FFTpow = FFTpow / (n * n)
    return FFTpow

def obtener_caracteristicas_espectrales(v, fm):
    n = len(v)
    vMean = np.mean(v)
    vFFT = v - vMean
    vFFT = vFFT * np.hanning(n)
    
    # Realizar FFT
    vFFT = np.fft.rfft(vFFT)
    vFFTpow = simur_getFFTpower(vFFT)
    
    # Encontrar las frecuencias dominantes
    FFTinterval = fm / (1.0 * n)  # Resolución en Hz
    f1_idx = np.argmax(vFFTpow)   # Índice del máximo de potencia
    p1 = vFFTpow[f1_idx]          # Potencia máxima
    f1 = f1_idx * FFTinterval     # Frecuencia en Hz
    
    # Descartamos el primer pico para encontrar el siguiente
    vFFTpow[f1_idx] = 0  
    f2_idx = np.argmax(vFFTpow)  # Índice del segundo máximo de potencia
    p2 = vFFTpow[f2_idx]         # Potencia del segundo pico
    f2 = f2_idx * FFTinterval    # Frecuencia en Hz
    
    # Cálculo de la entropía espectral
    vFFTpowsum = np.sum(vFFTpow)                                # Suma total de las potencias FFT
    p = vFFTpow / (vFFTpowsum + 1e-8)                           # Probabilidades normalizadas
    spectralEntropy = np.sum(-p * np.log10(p + 1E-8))           # Entropía espectral
    spectralEntropy = spectralEntropy / np.log10(len(vFFTpow))  # Normalizamos la entropía
    
    return [f1, p1, f2, p2, spectralEntropy], vFFTpow