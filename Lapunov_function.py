import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import signal
import sys

L = 0.1
Vin = 8
R = 5
C = 0.01

Tp = 0.001
Tf = 1
samples = int(Tf / Tp + 1)

T = np.linspace(0, Tf, samples)
x_1 = np.linspace(-20, 20, samples)
x_2 = np.linspace(-20, 20, samples)
#V = np.linspace(0, 0, samples)
#dV = np.linspace(0, 0, samples)

B = np.array([[x_1],
              [x_2]])

U = np.full(samples, 0.5)

A = np.array([[0, 1],
              [-1 / (C * L), -1 / (C * R)]])

B = np.array([[0],
              [Vin / (C * L)]])

C = np.array([[1, 0]])

D = np.array([[0]])

# symuluj układ
res = signal.lsim([A, B, C, D], U, T)
X = res[2]
Y = res[1]

Q = np.array([[2, -1],
              [-1, 2]])

# ujemna macierz Q, ze względu na definicję funkcji
Q1 = np.array([[-2, 1],
               [1, -2]])

H = np.array([[0, 1],
              [-100, -20]])

P = scipy.linalg.solve_continuous_lyapunov(H, Q1)

def V(x_1, x_2):
    return 0.5 * (0.2405 * x_1 * x_1 - 2 * x_1 * x_2 + 5.05 * x_2 * x_2)

X_1, X_2 = np.meshgrid(x_1, x_2)
Z = V(X_1, X_2)

"""
for i in range(samples):
    V[i] = 0.5 * (0.2405 * x_1[i] * x_1[i] - 2 * x_1[i] * x_2[i] + 5.05 * x_2[i] * x_2[i])
    dV[i] = - (x_1[i] * x_1[i] - x_1[i] * x_2[i] + x_2[i] * x_2[i])
    """

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X_1, X_2, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
plt.show()

