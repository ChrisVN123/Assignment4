import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize

df = pd.read_csv("transformer_data.csv")
matrix = df.to_numpy()
u = matrix[:,2:] #we only want the last three columns

def unpack_parameters(theta):
    A = theta[0]
    B = np.array(theta[1:4])  # 3 elements
    C = theta[4]
    Q = theta[5]  # Σ₁
    R = theta[6]  # Σ₂
    x0 = theta[7]  # initial latent state
    return A, B, C, Q, R, x0


def kalman_filter(y, u, theta):
    A, B, C, Q, R, x0 = unpack_parameters(theta)
    n = len(y)

    x_pred = np.zeros(n)
    P_pred = np.zeros(n)
    x_filt = np.zeros(n)
    P_filt = np.zeros(n)
    log_likelihoods = np.zeros(n)

    # Initialize
    x_filt[0] = x0
    P_filt[0] = 1.0  # You could also estimate this

    for t in range(1, n):
        # Predict
        x_pred[t] = A * x_filt[t-1] + np.dot(B, u[t-1])
        P_pred[t] = A**2 * P_filt[t-1] + Q

        # Update
        y_pred = C * x_pred[t]
        innovation = y[t] - y_pred
        S_t = C**2 * P_pred[t] + R
        K_t = P_pred[t] * C / S_t

        x_filt[t] = x_pred[t] + K_t * innovation
        P_filt[t] = (1 - K_t * C) * P_pred[t]

        # Log-likelihood
        log_likelihoods[t] = -0.5 * (np.log(2 * np.pi) + np.log(S_t) + innovation**2 / S_t)

    return log_likelihoods, x_filt

def negative_log_likelihood(theta, y, u):
    log_likelihoods, _ = kalman_filter(y, u, theta)
    return -np.sum(log_likelihoods)


def estimate_model(y, u):
    # Starting values: A, B1, B2, B3, C, Q, R, x0
    initial_guess = np.array([0.9, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 20.0])
    lower_bounds = [-1.5] * 5 + [1e-6, 1e-6, -np.inf]
    upper_bounds = [1.5] * 5 + [np.inf, np.inf, np.inf]

    result = minimize(negative_log_likelihood, initial_guess,
                      args=(y, u), bounds=list(zip(lower_bounds, upper_bounds)))

    return result


Y = df["Y"].values
result = estimate_model(Y, u)

print("Estimated parameters:", result.x)
log_lik, x_filtered = kalman_filter(Y, u, result.x)

# Optional plot
plt.plot(Y, label="Observed")
plt.plot(x_filtered, label="Filtered state")
plt.legend()
plt.show()
