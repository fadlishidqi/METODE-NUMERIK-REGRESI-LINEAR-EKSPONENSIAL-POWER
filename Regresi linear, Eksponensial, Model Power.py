import numpy as np
import matplotlib.pyplot as plt

# Data tegangan (x) dan waktu patah (y)
x = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y = np.array([40, 30, 25, 40, 18, 20, 22, 15])

def divided_differences(x_points, y_points):
    n = len(x_points)
    coef = np.zeros([n, n])
    coef[:, 0] = y_points
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (x_points[i + j] - x_points[i])
    return coef[0, :]

def newton_polynomial(coef, x_points, x):
    n = len(coef)
    result = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_points[j])
        result += term
    return result

# Hitung koefisien polinom Newton
coef = divided_differences(x, y)

# Contoh penggunaan: menghitung nilai interpolasi pada tegangan tertentu
x_targets = [8, 18, 28, 38]  # contoh tegangan yang ingin diinterpolasi
for x_target in x_targets:
    y_interpolated = newton_polynomial(coef, x, x_target)
    print(f"Waktu patah pada tegangan {x_target} kg/mm² adalah sekitar {y_interpolated:.2f} jam")

# Plot data asli
plt.scatter(x, y, color='green', label='Data asli')

# Plot hasil interpolasi
x_plot = np.linspace(min(x), max(x), 500)
y_plot = [newton_polynomial(coef, x, xi) for xi in x_plot]
plt.plot(x_plot, y_plot, label='Polinom Interpolasi Newton')

plt.xlabel('Tegangan (kg/mm²)')
plt.ylabel('Waktu patah (jam)')
plt.legend()
plt.title('Interpolasi Polinom Newton')
plt.show()

# Menghitung nilai interpolasi untuk nilai tegangan tertentu dan membandingkan dengan nilai asli
test_points = [5, 10, 15, 20, 25, 30, 35, 40, 8, 18, 28, 38]  # termasuk nilai-nilai dalam data asli dan nilai baru
for tp in test_points:
    interp_value = newton_polynomial(coef, x, tp)
    print(f"Interpolasi waktu patah pada tegangan {tp} kg/mm² adalah sekitar {interp_value:.2f} jam")
