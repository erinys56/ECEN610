import numpy as np
import matplotlib.pyplot as plt

Cs = 15.925e-12  # F (15.925 pF)
N = 8
f_clock = 2.4e9  # 2.4 GHz
frequencies = np.logspace(6, 10, 1000)  # 1 MHz ~ 10 GHz
omega = 2 * np.pi * frequencies


# Discharged after each N-cycle
H_discharge = ((1 - np.exp(-1j * 2 * np.pi * frequencies * N / f_clock)) /
               (1 - np.exp(-1j * 2 * np.pi * frequencies / f_clock))) * \
               (1 / (Cs * 1j * 2 * np.pi * frequencies))

# Never discharged (accumulated)
H_no_discharge = (1 / (Cs * 1j * 2 * np.pi * frequencies)) * \
                 (1 / (1 - np.exp(-1j * 2 * np.pi * frequencies * N / f_clock)))

# Plot - Discharged
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies, 20 * np.log10(np.abs(H_discharge)), label='Discharged')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function (Discharged Every N Cycles)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot - Not Discharged
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies, 20 * np.log10(np.abs(H_no_discharge)), label='Never Discharged', color='orange')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function (Never Discharged)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
