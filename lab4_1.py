import numpy as np
import matplotlib.pyplot as plt


f_in = 1e9            
f_s = 10e9            
tau = 10e-12          
T_s = 1 / f_s         
duration = 5e-9       
dt = 1e-12            
t = np.arange(0, duration, dt)


vin = np.sin(2 * np.pi * f_in * t)


vout = np.zeros_like(t)
vprev = 0
for i in range(1, len(t)):
    time_in_period = (t[i] % T_s)
    if time_in_period < T_s / 2:
        
        dv = (vin[i] - vprev) * dt / tau
        vprev += dv
    
    vout[i] = vprev


plt.figure(figsize=(10, 4))
plt.plot(t * 1e9, vin, label='Input (Vin)', linestyle='--')
plt.plot(t * 1e9, vout, label='Sampled Output (Vout)')
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (V)')
plt.title('ZOH Sampling Circuit Output')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
