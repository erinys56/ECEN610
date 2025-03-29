import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


fs = 10e9                    
Ts = 1 / fs                  
N = 1000                    
t = np.arange(N) * Ts        


frequencies = [0.2e9, 0.58e9, 1.0e9, 1.7e9, 2.4e9]
amplitudes = [0.1] * len(frequencies)
x_ideal = sum(A * np.sin(2 * np.pi * f * t) for A, f in zip(amplitudes, frequencies))


tau = 12e-12                 # 시간 상수 (12 ps)
Ton = Ts / 2                # 스위치 ON 시간 (50 ps)
alpha = 1 - np.exp(-Ton / tau)
y_sampled = alpha * x_ideal


LSB = 1 / 128
q_y = np.round(y_sampled / LSB) * LSB


E = q_y - x_ideal


min_M = 2
max_M = 10
ratios = []

for M in range(min_M, max_M + 1):
    X = []
    y = []

    for n in range(M, len(q_y)):
        X.append(q_y[n - M + 1:n])   # 과거 M-1개의 ADC 출력
        y.append(E[n])               # 현재 에러 (타깃)

    X = np.array(X)
    y = np.array(y)


    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    e_hat = model.predict(X)


    y_corrected = q_y[M:] + e_hat


    E_corrected = y_corrected - x_ideal[M:]
    var_E_corrected = np.var(E_corrected)


    sigma_q2 = LSB**2 / 12


    ratios.append(var_E_corrected / sigma_q2)


plt.figure(figsize=(8, 4))
plt.plot(range(min_M, max_M + 1), ratios, marker='o')
plt.xlabel('FIR Filter Length (M)')
plt.ylabel('Variance Ratio (Corrected Error / Quantization Noise)')
plt.title('Sampling Error Correction with FIR Filter')
plt.grid(True)
plt.tight_layout()
plt.show()
