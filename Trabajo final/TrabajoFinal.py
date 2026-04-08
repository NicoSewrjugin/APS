import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy import signal

# descargar datos
ticker = "SPY"

data = yf.download(
    ticker,
    start="2000-01-01",
    end="2025-01-01"
)

# precio de cierre
price = data["Close"]

# retornos logarítmicos
returns = np.log(price / price.shift(1))
returns = returns.dropna()

#%%
plt.figure(figsize=(12,6))

plt.plot(price)

plt.title("Precio del SPY (2000-2025)")
plt.xlabel("Tiempo[d]")
plt.ylabel("Precio[USD]")

plt.grid()
plt.show()

#%%
plt.figure(figsize=(12,6))

plt.plot(returns)

plt.title("Retornos logarítmicos")
plt.xlabel("Tiempo[d]")
plt.ylabel("Retorno[%]")

plt.grid()
plt.show()

#%%
plt.figure(figsize=(8,5))

plt.hist(returns, bins=100)

plt.title("Distribución de retornos")
plt.xlabel("Retorno[%]")
plt.ylabel("Frecuencia[Hz]")

plt.show()

#%%
ma20 = price.rolling(20).mean()
ma50 = price.rolling(50).mean()

plt.figure(figsize=(12,6))

plt.plot(price, label="Precio")
plt.plot(ma20, label="MA20")
plt.plot(ma50, label="MA50")

plt.title("Precio con medias móviles")
plt.legend()

plt.xlabel("Tiempo[d]")
plt.ylabel("Precio[USD]")

plt.grid()
plt.show()


zoom_days = 300 

plt.figure(figsize=(12,6))

# Usamos .iloc[-zoom_days:] para tomar solo el final de la serie
plt.plot(price.iloc[-zoom_days:], label="Precio") # alpha para que resalte la MA
plt.plot(ma20.iloc[-zoom_days:], label="MA20 (Rápida)", linewidth=2)
plt.plot(ma50.iloc[-zoom_days:], label="MA50 (Lenta)", linewidth=2)

plt.title(f"Zoom: Precio con Medias Móviles (Últimos {zoom_days} días)")
plt.legend()
plt.xlabel("Tiempo[d]")
plt.ylabel("Precio[USD]")
plt.grid(True, which='both', linestyle='--')
plt.show()

#%%
# señal
signal = returns.to_numpy().flatten()

# cantidad de datos
N = len(signal)

# FFT
fft_values = fft(signal)

# frecuencias
freqs = fftfreq(N, d=1)

# nos quedamos con la mitad positiva
mask = freqs > 0

freqs = freqs[mask]
fft_values = np.abs(fft_values[mask])

plt.figure(figsize=(10,5))

plt.plot(freqs, fft_values)

plt.title("Espectro de Fourier de los retornos")
plt.xlabel("Frecuencia[ciclos/día]")
plt.ylabel("Magnitud")

plt.grid()
plt.show()

#%%
#Welch

freqs_psd, psd = welch(signal, nperseg=256)

plt.figure(figsize=(10,5))

plt.semilogy(freqs_psd, psd)

plt.title("Densidad espectral de potencia")
plt.xlabel("Frecuencia[ciclos/día]")
plt.ylabel("PSD")

plt.grid()
plt.show()

#%%
#FFT de returns al cuadrado

# señal de retornos
signal = returns.to_numpy().flatten()

# retornos al cuadrado
signal_sq = signal**2

# cantidad de datos
N = len(signal_sq)

# FFT
fft_values_sq = fft(signal_sq)

# frecuencias
freqs_sq = fftfreq(N, d=1)

# mitad positiva
mask = freqs_sq > 0

freqs_sq = freqs_sq[mask]
fft_values_sq = np.abs(fft_values_sq[mask])

plt.figure(figsize=(10,5))
plt.plot(freqs_sq, fft_values_sq)

plt.title("Espectro de Fourier de los retornos al cuadrado")
plt.xlabel("Frecuencia[ciclos/día]")
plt.ylabel("Magnitud")

plt.grid()
plt.show()

#%%
#Welch de returns al cuadrado
freqs_psd_sq, psd_sq = welch(signal_sq, nperseg=512, noverlap=256)

plt.figure(figsize=(10,5))

plt.loglog(freqs_psd_sq, psd_sq)

plt.title("Densidad espectral de potencia (retornos al cuadrado)")
plt.xlabel("Frecuencia [ciclos/día]")
plt.ylabel("PSD")

plt.grid()
plt.show()

#%%
#Medias moviles

# eliminar NaN
ma20 = ma20.dropna()
ma50 = ma50.dropna()

# retornos de las medias móviles
returns_ma20 = np.log(ma20 / ma20.shift(1)).dropna()
returns_ma50 = np.log(ma50 / ma50.shift(1)).dropna()

# convertir a vectores 1D
signal_original = returns.to_numpy().flatten()
signal_ma20 = returns_ma20.to_numpy().flatten()
signal_ma50 = returns_ma50.to_numpy().flatten()

# señal original
freqs, psd_original = welch(signal_original, nperseg=256)

# señal filtrada MA20
freqs20, psd_ma20 = welch(signal_ma20, nperseg=256)

# señal filtrada MA50
freqs50, psd_ma50 = welch(signal_ma50, nperseg=256)

plt.figure(figsize=(10,6))

plt.semilogy(freqs, psd_original, label="Original")
plt.semilogy(freqs20, psd_ma20, label="Media móvil 20")
plt.semilogy(freqs50, psd_ma50, label="Media móvil 50")

plt.title("Comparación de densidad espectral de potencia")
plt.xlabel("Frecuencia[ciclos/día]")
plt.ylabel("PSD")

plt.legend()
plt.grid()

plt.show()

#%%

#Filtro Butter

# --- Configuración del Filtro Butterworth ---
fs = 1.0  # Frecuencia de muestreo: 1 muestra por día
# Queremos una frecuencia de corte similar a una media móvil de 20 días.
# F_corte = 1 / Periodo = 1 / 20 = 0.05 Hz. 
fc = 0.05 
# Queremos una frecuencia de corte similar a una media móvil de 50 días.
# F_corte = 1 / Periodo = 1 / 50 = 0.02 Hz. 
fc2 = 0.02

# b, a son los coeficientes del filtro
b, a = signal.butter(N=2, Wn=fc, btype='low', fs=fs)
b2, a2 = signal.butter(N=2, Wn=fc2, btype='low', fs=fs)

price_clean = price.dropna()
# Convertimos a numpy array explícitamente para evitar problemas de formato
price_array = price_clean.values.flatten()
    
# Aplicamos el filtro
price_butter_all = signal.filtfilt(b, a, price_array)
price_butter_all2 = signal.filtfilt(b2, a2, price_array)
    
# Volvemos a convertir a Serie de Pandas con el mismo índice
price_butter = pd.Series(price_butter_all, index=price_clean.index)
price_butter2 = pd.Series(price_butter_all2, index=price_clean.index)
    
# Gráfico de Zoom
zoom_days = 300

plt.figure(figsize=(12,6))
plt.plot(price_clean.iloc[-zoom_days:], label="Precio Original")
plt.plot(price_butter.iloc[-zoom_days:], label="Butterworth (IIR - Fase Cero) Fc=0.05Hz", linewidth=2, color='red')
plt.plot(price_butter2.iloc[-zoom_days:], label="Butterworth (IIR - Fase Cero) Fc=0.02Hz", linewidth=2, color='green')

plt.title(f"Zoom: Precio con Filtros Butterworth (Últimos {zoom_days} días)")
plt.legend()
plt.xlabel("Tiempo[d]")
plt.ylabel("Precio[USD]")
plt.grid()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(price_clean.iloc[-zoom_days:], label="Precio Original")
plt.plot(ma20.loc[price_clean.index].iloc[-zoom_days:], label="MA20 (FIR)", ls='--', color='red')
plt.plot(ma50.loc[price_clean.index].iloc[-zoom_days:], label="MA50 (FIR)", ls='--', color='green')
plt.plot(price_butter.iloc[-zoom_days:], label="Butterworth (IIR - Fase Cero) Fc=0.05Hz", linewidth=2, color='red')
plt.plot(price_butter2.iloc[-zoom_days:], label="Butterworth (IIR - Fase Cero) Fc=0.02Hz", linewidth=2, color='green')

plt.title(f"Comparación Técnica: FIR vs IIR (Zoom {zoom_days} días)")
plt.legend()
plt.xlabel("Tiempo[d]")
plt.ylabel("Precio[USD]")
plt.grid()
plt.show()

# Compensamos el retraso de fase desplazando las series hacia atrás
# MA20: retraso de 10 días aprox.
ma20_aligned = ma20.shift(-10)

# MA50: retraso de 25 días aprox.
ma50_aligned = ma50.shift(-25)

plt.figure(figsize=(12,6))
plt.plot(price_clean.iloc[-zoom_days:], label="Precio Original")
plt.plot(ma20_aligned.loc[price_clean.index].iloc[-zoom_days:], label="MA20 alineada (FIR)", ls='--', color='red')
plt.plot(ma50_aligned.loc[price_clean.index].iloc[-zoom_days:], label="MA50 alineada (FIR)", ls='--', color='green')
plt.plot(price_butter.iloc[-zoom_days:], label="Butterworth (IIR - Fase Cero) Fc=0.05Hz", linewidth=2, color='red')
plt.plot(price_butter2.iloc[-zoom_days:], label="Butterworth (IIR - Fase Cero) Fc=0.02Hz", linewidth=2, color='green')

plt.title(f"Comparación Técnica 2: FIR vs IIR (Zoom {zoom_days} días) con medias móviles alineadas")
plt.legend()
plt.xlabel("Tiempo[d]")
plt.ylabel("Precio[USD]")
plt.grid()
plt.show()