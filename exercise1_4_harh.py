import numpy as np
import matplotlib.pyplot as plt
from exercise1_2 import simulated
from scipy.optimize import minimize

def myKalmanFilter(y, theta, R, X0, P0):
    a, b, Q = theta
    n = len(y)

    # Allocate arrays
    x_pred = np.zeros(n)  # X_hat_{t|t-1}
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)  # X_hat_{t|t}
    P_filt = np.zeros(n)
    innovations = np.zeros(n)
    S = np.zeros(n)       # Innovation variance

    # Initialization
    x_filt[0] = X0
    P_filt[0] = P0
    L=0
    for t in range(1, n):
        # Predict
        x_pred[t] = a * x_filt[t-1] + b
        P_pred[t] = a**2 * P_filt[t-1] + Q

        # Innovation
        innovations[t] = y[t] - x_pred[t]
        S[t] = P_pred[t] + R

        # Kalman Gain
        K_t = P_pred[t] / S[t]

        # Update
        x_filt[t] = x_pred[t] + K_t * innovations[t]
        P_filt[t] = (1 - K_t) * P_pred[t]
        L += -1/2*np.sum(np.log(S[t])+innovations[t]**2*(S[t]**(-1)))

    return x_pred, P_pred, innovations, S, x_filt, P_filt, -L


n=100
X0 = 5
P0 = 1
theta = [1, 0.9, 1]  # a, b, Q
R = 1  # observation noise variance
X, Y = simulated()
# Apply Kalman filter to observations Y_t

def objective(th):
    _, _, _, _, _, _, L_val = myKalmanFilter(Y, th, R, X0, P0)
    return L_val

result = minimize(objective, x0=theta, method='L-BFGS-B', bounds=[(0.1, 0.9), (0.1, 10), (0.1, 10)])

print("Optimized parameters:", result.x)
print("Log-likelihood:", result.fun)
# Plotting the results
