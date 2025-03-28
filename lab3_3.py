import numpy as np
import matplotlib.pyplot as plt

# Parameters
CR = 15.925e-12  # Rotating capacitor value (Farads)
N = 8            # Number of clock cycles per capacitor
f_clock = 2.4e9  # Clock frequency (Hz)
TN = N / f_clock  # Duration for one capacitor

# Frequency axis (1 MHz ~ 10 GHz)
frequencies = np.logspace(6, 10, 1000)
omega = 2 * np.pi * frequencies

# Transfer function H(f) for 4 interleaved capacitors (1 bank)
numerator = 1 - np.exp(-1j * 2 * np.pi * frequencies * 4 * TN)
denominator = 1 - np.exp(-1j * 2 * np.pi * frequencies * TN)
H = numerator / (CR * 1j * 2 * np.pi * frequencies * denominator)

# Plotting
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies, 20 * np.log10(np.abs(H)), label='Discharged (8 total caps, 4 per bank)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function - Problem 3(a) Discharged Case')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
 

# Transfer function H(f) - Never Discharged case
numerator = 1 - np.exp(-1j * 2 * np.pi * frequencies * 4 * TN)
denominator = (1 - np.exp(-1j * 2 * np.pi * frequencies * TN)) ** 2
H = numerator / (CR * 1j * 2 * np.pi * frequencies * denominator)

# Plot
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies, 20 * np.log10(np.abs(H)), label='Never Discharged (4 caps)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function - Problem 3(b) Never Discharged Case')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Capacitor values for CR1 ~ CR4 (Farads)
CR1 = 14e-12
CR2 = 16e-12
CR3 = 12e-12
CR4 = 20e-12

CR_list = [CR1, CR2, CR3, CR4]

# Clock parameters
N = 8
f_clock = 2.4e9
TN = N / f_clock

# Frequency axis
frequencies = np.logspace(6, 10, 1000)
omega = 2 * np.pi * frequencies

# Compute H(f)
numerator = 1 - np.exp(-1j * 2 * np.pi * frequencies * TN)
denominator = 1j * 2 * np.pi * frequencies

sum_terms = np.zeros_like(frequencies, dtype=complex)
for k in range(4):
    delay = np.exp(-1j * 2 * np.pi * frequencies * k * TN)
    sum_terms += delay / CR_list[k]

H = numerator / denominator * sum_terms

# Plot
plt.figure(figsize=(10, 5))
plt.semilogx(frequencies, 20 * np.log10(np.abs(H)), label='Discharged, Unequal $C_{Rk}$')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function - Problem 3(c): Unequal Capacitor Sizes')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
