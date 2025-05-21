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
    return A, B, C, Q, R, x0

def kalman_filter(y, u, theta):
    A, B, C, Q, R, x0 = unpack_parameters(theta)
    n = len(y)

    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    log_likelihoods = np.zeros(n-1)
    innov   = np.zeros(n)
    S_t = np.zeros(n)
    x_filt[0] = x0
    P_filt[0] = np.eye(2)

    for t in range(1, n):
        # Predict
        x_pred[t] = A @ x_filt[t-1] + B @ u[t-1]
        P_pred[t] = A @ P_filt[t-1] @ A.T + Q

        # Update
        y_pred = C @ x_pred[t]
        innov[t] = (y[t] - y_pred).item()
        S_t[t] = ((C @ P_pred[t] @ C.T) + R).item()
        K_t = P_pred[t] @ C.T / S_t[t]

        x_filt[t] = x_pred[t] + (K_t.flatten() * innov[t])
        P_filt[t] = (np.eye(2) - K_t @ C) @ P_pred[t]

        log_likelihoods[t-1] = -0.5 * (np.log(2 * np.pi) + np.log(S_t[t]) + (innov[t] ** 2) / S_t[t])
    log_likelihood = np.sum(log_likelihoods)

    return x_pred, P_pred, x_filt, P_filt, innov, S_t, log_likelihood

def negative_log_likelihood(theta, y, u):
    *_, log_likelihood = kalman_filter(y, u, theta)
    return -log_likelihood

def estimate_model(y, u):
    # Initial values
    theta0 = np.array([
        1, 0.0, 0.0, 1,            # A (flattened)
        0.1, 0.0, 0.0,                 # B row 1
        0.0, 0.1, 0.0,                 # B row 2
        1, 0,                      # C
        1, 1,      # Q diag (log for positivity)
        1,                   # R (log for positivity)
        20.0, 20.0                      # x0
    ])
    bounds = [(-2,2)]*4 + [(-5,5)]*6 + [(-5,5)]*2 + [(None,None)]*5
    result = minimize(negative_log_likelihood, theta0, args=(y, u),bounds = bounds,
                      method="L-BFGS-B", options={'maxiter': 5000})
    return result


# Estimate
result = estimate_model(Y, u)
theta_hat = result.x
print("Estimated parameters:", theta_hat)

# Predict
A, B, C, Q, R, x0 = unpack_parameters(theta_hat)
# run filter once more with the MLE
x_p, P_p, x_filt, _, innov, _, _= kalman_filter(Y, u, theta_hat)

# point forecast of yₜ
Y_pred = (C @ x_p[1:].T).ravel()  
residuals = Y[1:] - Y_pred
stderr = np.sqrt((C @ P_p[1:] @ C.T).ravel() + R)
Y_pred_lower = Y_pred-1.96*stderr
Y_pred_higher  =  Y_pred+1.96*stderr

# ------------------------------
# Plotting
# ------------------------------
#Simple conf interval with true y
plt.plot(Y_pred_lower, label = "Conf interval", color = "black", linestyle='-',linewidth=0.5)   
plt.plot(Y_pred_higher, color = "black", linestyle='-',linewidth=0.5)
plt.plot(df["Y"][1:], label="True $y_t$")
plt.plot(Y_pred, label="$\\hat{Y}_{t+1|t}$", linestyle='-.')
plt.legend()
plt.title("Kalman Filtering Results (2d)")
plt.savefig(f"images/2.3/ex2_3_confidence_int_prediction.png")
plt.clf()


###3

# Combined plot for residual diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residual time series
axes[0, 0].plot(residuals)
axes[0, 0].set_title("Residuals")
axes[0, 0].grid()

# ACF
plot_acf(innov[1:], lags=40, ax=axes[0, 1])
axes[0, 1].set_title("ACF of Residuals")

# PACF
plot_pacf(innov[1:], lags=40, ax=axes[1, 0])
axes[1, 0].set_title("PACF of Residuals")

# QQ-plot
stats.probplot(innov[1:], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("QQ-plot of Residuals")
axes[1, 1].grid()

plt.tight_layout()
plt.savefig("images/2.3/ex2_3_stats.png")


# 2) AIC and BIC
n = len(Y)
k = len(theta_hat)
lnL =LL = -negative_log_likelihood(theta_hat, Y, u)


AIC = 2*k - 2*lnL
BIC = k*np.log(n) - 2*lnL

print(f"AIC = {AIC:.2f}")
print(f"BIC = {BIC:.2f}")

# (Optionally save to file or DataFrame)
info = pd.DataFrame({
    "stat": ["n_obs","n_params","logLik","AIC","BIC"],
    "value": [n, k, lnL, AIC, BIC]
})
info.to_csv(f"images/2.3/info_criteria.csv", index=False)



params = pd.DataFrame({
    "params": ["A", "B", "C", "Q", "R", "x0",],
    "value": [A, B, C, Q, R, x0]
})
params.to_csv(f"images/2.3/params.csv", index=False)