import numpy as np
import matplotlib.pyplot as plt


CH = 15.425e-12  # History capacitor
CR = 0.5e-12     # Rotating capacitor


Ceff = CH + CR
a1 = CH / Ceff  


N = 8
f_clock = 2.4e9  
TN = N / f_clock  

frequencies = np.logspace(6, 10, 1000)
omega = 2 * np.pi * frequencies

numerator = 1
denominator = a1*Ceff * 1j * 2 * np.pi * frequencies * (1 - np.exp(-1j * 2 * np.pi * frequencies * TN))
H = numerator / denominator

plt.figure(figsize=(10, 5))
plt.semilogx(frequencies, 20 * np.log10(np.abs(H)), label=f'a₁ = {a1:.4f}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function with CH, CR (Never Discharged)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
