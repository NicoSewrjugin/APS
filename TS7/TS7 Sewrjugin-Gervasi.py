# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 19:13:24 2025

@author: Sofía
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_freq_response(num, den):
    zeros, poles, _ = signal.tf2zpk(num, den)
    w, h = signal.freqz(num, den, worN=8000)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Mapa de polos y ceros
    axs[0].plot(np.real(zeros), np.imag(zeros), 'o', label='Ceros', markersize=10)
    axs[0].plot(np.real(poles), np.imag(poles), 'x', label='Polos', markersize=12)
    axs[0].axhline(0, color='gray', lw=1)
    axs[0].axvline(0, color='gray', lw=1)
    axs[0].set_title("Mapa de polos y ceros")
    axs[0].set_xlabel(r'Re{z}')
    axs[0].set_ylabel(r'Im{z}')
    axs[0].legend()

    # Dibujar círculo unidad
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
    axs[0].add_artist(circle)
    axs[0].set_xlim(-1.5, 1.5)
    axs[0].set_ylim(-1.5, 1.5)
    axs[0].set_aspect('equal', adjustable='box')

    # Respuesta en módulo
    axs[1].plot(w, np.abs(h), 'b')
    axs[1].set_title("Respuesta en módulo")
    axs[1].set_xlabel(r'Frecuencia $\omega$ [rad/muestra]')
    axs[1].set_ylabel(r'$|H(e^{j\omega})|$')
    axs[1].grid(True)
    axs[1].set_xticks([0, np.pi/2, np.pi])
    axs[1].set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

    # Respuesta en fase
    axs[2].plot(w, np.angle(h), 'r')
    axs[2].set_title("Respuesta en fase")
    axs[2].set_xlabel(r'Frecuencia $\omega$ [rad/muestra]')
    axs[2].set_ylabel(r'$\angle H(e^{j\omega})$ [rad]')
    axs[2].grid(True)
    axs[2].set_xticks([0, np.pi/2, np.pi])
    axs[2].set_xticklabels([r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

#%% A
num = [1, 1, 1, 1]  # 1 + z + z^2 + z^3 
den = [1, 0, 0, 0]  # z^3 

plot_freq_response(num, den)

#%% B

num_b = [1, 1, 1, 1, 1]  # 1 + z + z^2 + z^3 + z^4
den_b = [1, 0, 0, 0, 0]  # z^4

plot_freq_response(num_b, den_b)

#%% C

num_c = [1, -1]  # -1 + z
den_c = [1, 0]      # z

plot_freq_response(num_c, den_c)

#%% D

num_d = [1, 0, -1]  # -1 + z^2
den_d = [1, 0, 0]         

plot_freq_response(num_d, den_d)

