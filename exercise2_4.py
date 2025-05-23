import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats
from matplotlib.pyplot import cm


# Load data
df = pd.read_csv("transformer_data.csv")
matrix = df.to_numpy()
u = matrix[:, 2:]  # 3 exogenous inputs
Y = df["Y"].values

# ------------------------------
# 2D Kalman Filter Setup
# ------------------------------
def unpack_parameters(theta):
    A = theta[0:4].reshape(2, 2)          # A: 2×2
    B = theta[4:10].reshape(2, 3)         # B: 2×3
    Q = np.diag(np.exp(theta[10:12]))     # Q: diag, positiveness via exp
    R = np.exp(theta[12])                 # R: scalar, positive
    x0 = theta[13:15]                     # initial state (2,)
    P0 = theta[15:19].reshape(2, 2)       # initial covariance (2×2)
    return A, B, Q, R, x0, P0

def kalman_filter(y, u, theta,C):
    A, B, Q, R, x0, P0 = unpack_parameters(theta)
    n = len(y)
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    log_likelihoods = np.zeros(n-1)
    innov   = np.zeros(n)
    S_t = np.zeros(n)
    x_filt[0] = x0
    P_filt[0] = P0

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

def negative_log_likelihood(theta, y, u, c):
    *_, log_likelihood = kalman_filter(y, u, theta, c)
    return -log_likelihood

def estimate_model(y, u):
    # Initial values
    theta0 = np.array([
        1, 0.0, 0.0, 1,             # A (4)
        0.1, 0.0, 0.0,              # B row 1 (3)
        0.0, 0.1, 0.0,              # B row 2 (3) → total B = 6
        1, 1,                       # log‑diag(Q)  (2)
        1,                          # log(R)       (1)
        23.5, 23.5,                 # x0           (2)
        1, 0, 0, 1                  # P0 (flattened 2×2, start = I)
    ])
    C = np.array([1, 0]).reshape(1, 2)  
    bounds = (
        [(-2, 2)] * 4 +            # A
        [(-5, 5)] * 6 +            # B
        [(None, None)] * 2 +       # log‑diag(Q)
        [(None, None)] +           # log(R)
        [(None, None)] * 2 +       # x0
        [(1e-8, 100)] * 4          # P0 entries
    )
    result = minimize(negative_log_likelihood, theta0, args=(y, u, C),bounds = bounds,
                      method="L-BFGS-B", options={'maxiter': 5000})
    return result


# Estimate
result = estimate_model(Y, u)
theta_hat = result.x
print("Estimated parameters:", theta_hat)

# Predict
A, B, Q, R, x0, P0 = unpack_parameters(theta_hat)
C = np.array([1, 0]).reshape(1, 2)  # C: 1x2
# run filter once more with the MLE
x_p, P_p, x_filt, _, innov, _, _= kalman_filter(Y, u, theta_hat, C)

# Estimate
result = estimate_model(Y, u)
theta_hat = result.x
print("Estimated parameters:", theta_hat)

# Predict
A, B, Q, R, x0, P0 = unpack_parameters(theta_hat)
C = np.array([1, 0]).reshape(1, 2)  # C: 1x2
# run filter once more with the MLE
x_p, P_p, x_filt, _, innov, _, _= kalman_filter(Y, u, theta_hat, C)

Y_pred = (C @ x_p[1:].T).ravel()  
residuals = Y[1:] - Y_pred
cols = cm.rainbow(np.linspace(0, 1, 3))   # three distinct colours

fig, ax1 = plt.subplots(figsize=(10, 5))

# -- latent states (primary axis) ---------------------------------------------
ax1.plot(df["time"], x_filt[:, 0], label=r"$x_{1,t}$")

# exogenous inputs Ta and S on the primary axis
ax1.plot(df["time"], df["Ta"], color=cols[0], ls="--", alpha=0.5,
         label=r"$T_{a,t}$")
ax1.plot(df["time"], df["I"],  color=cols[1], ls="--", alpha=0.5,
         label=r"$\Phi_{I,t}$")

ax1.set_xlabel("Time")
ax1.set_ylabel(r"$x_{1,t}$, $T_{a,t}$, $\Phi_{I,t}$ (primary axis)")                    
ax1.set_title("Kalman-filter state estimates and exogenous inputs")

# -- secondary axis for I ------------------------------------------------------
ax2 = ax1.twinx()
ax2.plot(df["time"], x_filt[:, 1], label=r"$x_{2,t}$",color = "orange")
ax2.plot(df["time"], df["S"], color=cols[2], ls="--", alpha=0.5,
         label=r"$\Phi_{s,t}$")
ax2.set_ylabel(r"$x_{2,t}$, $\Phi_{s,t}$ (secondary axis)")

# -- combined legend -----------------------------------------------------------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

fig.tight_layout()
fig.savefig("images/2.4/kalman_filter_state_estimates_w_exogenous.png")

plt.clf()

fig, ax1 = plt.subplots(figsize=(10, 5))
# -- visible state (primary axis) ---------------------------------------------
ax1.plot(df["time"], x_filt[:, 0], label=r"$x_{1,t}$")
ax1.set_xlabel("Time")
ax1.set_ylabel(r"$x_{1,t}$ (primary axis)")
ax1.set_title("Kalman-filter state estimates")
# -- secondary axis for latent state ------------------------------------------------------
ax2 = ax1.twinx()
ax2.plot(df["time"], x_filt[:, 1], label=r"$x_{2,t}$",color = "orange")
ax2.set_ylabel(r"$x_{2,t}$ (secondary axis)")
# -- combined legend -----------------------------------------------------------
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
fig.tight_layout()
plt.savefig("images/2.4/kalman_filter_state_estimates.png")



# Now we save Matrices A and B in csv format
A = pd.DataFrame(A)
A.to_csv("images/2.4/A.csv", index=False)
B = pd.DataFrame(B)
B.to_csv("images/2.4/B.csv", index=False)

params = pd.DataFrame({
    "params": ["A", "B", "C", "Q", "R", "x0","P0"],
    "value": [A, B, C, Q, R, x0,P0]
})
params.to_csv(f"images/2.4/params.csv", index=False)

# 2) AIC and BIC
n = len(Y)
k = len(theta_hat)
lnL =LL = -negative_log_likelihood(theta_hat, Y, u, C)


AIC = 2*k - 2*lnL
BIC = k*np.log(n) - 2*lnL
RMSE = np.sqrt(np.mean(residuals**2))
print(f"AIC = {AIC:.2f}")
print(f"BIC = {BIC:.2f}")
print(f"RMSE = {RMSE:.2f}")
# (Optionally save to file or DataFrame)
info = pd.DataFrame({
    "stat": ["n_obs","n_params","logLik","AIC","BIC", "RMSE"],
    "value": [n, k, lnL, AIC, BIC, RMSE]
})
info.to_csv(f"images/2.4/info_criteria.csv", index=False)

# # ------------------------------

# # point forecast of yₜ
# Y_pred = (C @ x_p.T).ravel() + e  
# residuals = Y - Y_pred
# stderr = np.sqrt((C @ P_p @ C.T).ravel() + R)
# Y_pred_lower = Y_pred-1.96*stderr
# Y_pred_higher  =  Y_pred+1.96*stderr

# # ------------------------------
# # Plotting
# # ------------------------------
# #Simple conf interval with true y
# plt.plot(Y_pred_lower, label = "Conf interval", color = "black", linestyle='-',linewidth=0.5)   
# plt.plot(Y_pred_higher, color = "black", linestyle='-',linewidth=0.5)
# plt.plot(df["Y"], label="True $y_t$")
# plt.plot(Y_pred, label="$\\hat{Y}_{t+1|t}$", linestyle='-.')
# plt.legend()
# plt.title("Kalman Filtering Results (2d)")
# plt.savefig(f"images/2.3/Conf_interval_1step.png")
# plt.clf()


# ###3

# # Combined plot for residual diagnostics
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# # Residual time series
# axes[0, 0].plot(residuals)
# axes[0, 0].set_title("Residuals")
# axes[0, 0].grid()

# # ACF
# plot_acf(innov[1:], lags=40, ax=axes[0, 1])
# axes[0, 1].set_title("ACF of Residuals")

# # PACF
# plot_pacf(innov[1:], lags=40, ax=axes[1, 0])
# axes[1, 0].set_title("PACF of Residuals")

# # QQ-plot
# stats.probplot(innov[1:], dist="norm", plot=axes[1, 1])
# axes[1, 1].set_title("QQ-plot of Residuals")
# axes[1, 1].grid()

# plt.tight_layout()
# plt.savefig("images/2.3/stats.png")


# # 2) AIC and BIC
# n = len(Y)
# k = len(theta_hat)
# lnL =LL = -negative_log_likelihood(theta_hat, Y, u)


# AIC = 2*k - 2*lnL
# BIC = k*np.log(n) - 2*lnL

# print(f"AIC = {AIC:.2f}")
# print(f"BIC = {BIC:.2f}")

# # (Optionally save to file or DataFrame)
# info = pd.DataFrame({
#     "stat": ["n_obs","n_params","logLik","AIC","BIC"],
#     "value": [n, k, lnL, AIC, BIC]
# })
# info.to_csv(f"images/2.3/info_criteria.csv", index=False)



# params = pd.DataFrame({
#     "params": ["A", "B", "C", "Q", "R", "x0",],
#     "value": [A, B, C, Q, R, x0]
# })
# params.to_csv(f"images/2.3/params.csv", index=False)