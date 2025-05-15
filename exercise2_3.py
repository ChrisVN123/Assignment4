import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats

# Load data
df = pd.read_csv("transformer_data.csv")
matrix = df.to_numpy()
u = matrix[:, 2:]  # 3 exogenous inputs
Y = df["Y"].values

# ------------------------------
# 2D Kalman Filter Setup
# ------------------------------
def unpack_parameters(theta):
    A = theta[0:4].reshape(2, 2)         # A: 2x2
    B = theta[4:10].reshape(2, 3)        # B: 2x3
    C = theta[10:12].reshape(1, 2)       # C: 1x2
    Q = np.diag(np.exp(theta[12:14]))    # Q: diagonal 2x2, exponentiated for positivity
    R = np.exp(theta[14])               # R: scalar, exponentiated
    x0 = theta[15:17]                   # Initial state: 2D
    e = theta[17]                       # Observation bias
    return A, B, C, Q, R, x0, e

def kalman_filter(y, u, theta):
    A, B, C, Q, R, x0, e = unpack_parameters(theta)
    n = len(y)

    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    log_likelihoods = np.zeros(n)

    x_filt[0] = x0
    P_filt[0] = np.eye(2)

    for t in range(1, n):
        # Predict
        x_pred[t] = A @ x_filt[t-1] + B @ u[t-1]
        P_pred[t] = A @ P_filt[t-1] @ A.T + Q

        # Update
        y_pred = C @ x_pred[t] + e
        innovation = y[t] - y_pred
        S_t = C @ P_pred[t] @ C.T + R
        K_t = P_pred[t] @ C.T / S_t

        x_filt[t] = x_pred[t] + (K_t @ innovation)
        P_filt[t] = P_pred[t] - K_t @ C @ P_pred[t]

        log_likelihoods[t] = -0.5 * (np.log(2 * np.pi) + np.log(S_t) + (innovation ** 2) / S_t)

    return log_likelihoods, x_filt

def negative_log_likelihood(theta, y, u):
    log_likelihoods, _ = kalman_filter(y, u, theta)
    return -np.sum(log_likelihoods)

def estimate_model(y, u):
    # Initial values
    theta0 = np.array([
        0.8, 0.0, 0.0, 0.7,            # A (flattened)
        0.1, 0.0, 0.0,                 # B row 1
        0.0, 0.1, 0.0,                 # B row 2
        1.0, 0.0,                      # C
        np.log(0.1), np.log(0.1),      # Q diag (log for positivity)
        np.log(0.1),                   # R (log for positivity)
        0.0, 0.0,                      # x0
        0.0                            # e
    ])

    result = minimize(negative_log_likelihood, theta0, args=(y, u),
                      method="L-BFGS-B", options={'maxiter': 500})
    return result

# Estimate
result = estimate_model(Y, u)
theta_hat = result.x
print("Estimated parameters:", theta_hat)

# Predict
log_lik, x_filtered = kalman_filter(Y, u, theta_hat)
A, B, C, Q, R, x0, e = unpack_parameters(theta_hat)
Y_pred = (C @ x_filtered.T).flatten() + e
residuals = Y - Y_pred

# ------------------------------
# Plotting
# ------------------------------


# Combined plot for residual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residual time series
axes[0, 0].plot(residuals)
axes[0, 0].set_title("Residuals")
axes[0, 0].grid()

# ACF
plot_acf(residuals, lags=40, ax=axes[0, 1])
axes[0, 1].set_title("ACF of Residuals")

# PACF
plot_pacf(residuals, lags=40, ax=axes[1, 0])
axes[1, 0].set_title("PACF of Residuals")

# QQ-plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("QQ-plot of Residuals")
axes[1, 1].grid()

plt.tight_layout()
plt.show()

# AIC and BIC
k = len(theta_hat)
n = len(Y)
LL = -negative_log_likelihood(theta_hat, Y, u)
AIC = 2 * k - 2 * LL
BIC = k * np.log(n) - 2 * LL

print(f"AIC: {AIC:.2f}, BIC: {BIC:.2f}")
