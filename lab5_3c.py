import numpy as np
import matplotlib.pyplot as plt

# 주어진 데이터
measured = np.array([-0.01, 0.105, 0.195, 0.28, 0.37, 0.48, 0.6, 0.75])
N = len(measured) - 1  # 3-bit → 8 levels → 7 intervals
LSB = 0.1  # 이상적인 LSB 크기

# End-point 보정 (ideal straight line 구하기)
x = np.arange(0, 8)
ideal_start = measured[0]
ideal_end = measured[-1]
ideal_line = ideal_start + (ideal_end - ideal_start) / N * x

# DNL 계산
ideal_steps = np.diff(ideal_line)
actual_steps = np.diff(measured)
dnl = (actual_steps - LSB) / LSB

# INL 계산
inl = (measured - ideal_line) / LSB

# 결과 출력
for i in range(N):
    print(f"Code {i+1}: DNL = {dnl[i]:.3f} LSB")

for i in range(8):
    print(f"Code {i}: INL = {inl[i]:.3f} LSB")

# DNL & INL 시각화 (선택)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.stem(range(1, 8), dnl, basefmt=" ")
plt.title("DNL")
plt.xlabel("Code")
plt.ylabel("DNL (LSB)")

plt.subplot(1, 2, 2)
plt.stem(range(8), inl, basefmt=" ")
plt.title("INL")
plt.xlabel("Code")
plt.ylabel("INL (LSB)")

plt.tight_layout()
plt.show()
