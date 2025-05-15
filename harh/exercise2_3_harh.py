from paths import PARENT_DIR
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

raw = pd.read_csv(f"{PARENT_DIR}/transformer_data.csv")

# ──────────────────────────────────────────────────────────────────────────────
# 2-dimensional state-space model
#   xₜ  ∈ ℝ²   (latent state: e.g. [core-temp, temp-drift])
#   yₜ  ∈ ℝ    (single measured output)
#
#   xₜ   =  A · xₜ₋₁  +  B · uₜ₋₁  +  wₜ        ,  wₜ ~  𝓝(0, Q)
#   yₜ   =  C · xₜ    +  vₜ                       ,  vₜ ~  𝓝(0, R)
#
# Dimensions
#   A : 2×2               (4 free elements)
#   B : 2×3  (Ta,S,I)     (6 free elements)
#   C : 1×2               (2 free elements)
#   Q : diag(q₁,q₂)       (2 free elements)   ←  keep diagonal to avoid over-
#   P0: diag(p₁,p₂)       (2 free elements)      parametrisation at first
#   R : scalar            (1 free element)
#   X0: 2-vector          (2 free elements)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np

def unpack(params):
    """Turn  *flat*  parameter vector -> individual matrices/vectors."""
    A  = params[0:4].reshape(2, 2)
    B  = params[4:10].reshape(2, 3)
    C  = params[10:12].reshape(1, 2)
    q1, q2          = params[12:14]
    r               = params[14]
    X0              = params[15:17]
    p1, p2          = params[17:19]
    Q  = np.diag([q1, q2])
    P0 = np.diag([p1, p2])
    return A, B, C, Q, r, X0, P0


def kf_logLik_2d(params, y, U):
    """
    2-D Kalman filter; returns   −log ℓ  (so we can minimise).
    """
    A, B, C, Q, R, x0, P_filt = unpack(params)
    n = len(y)

    # Pre-allocate
    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    innovations = np.zeros(n)
    S          = np.zeros(n)
    x_filt = np.zeros(n)  # X_hat_{t|t}
    x_filt[0] = x0
    ll = 0.0                                   # log-likelihood accumulator

    for t in range(1, n):
        # ───── Prediction step ───────────────────────────────────────────────
        x_pred[t]   = A @ x_filt[t-1]  +  B @ U[t-1]
        P_pred[t]   = A @ P_filt[t-1] @ A.T + Q

        # ───── Innovation (measurement residual) ────────────────────────────
        innovations[t] = y[t] - (C @ x_pred[t])[0]
        S[t]           = (C @ P_pred[t] @ C.T)[0, 0] + R

        # ───── Kalman gain ──────────────────────────────────────────────────
        K   = P_pred[t] @ C.T / S[t]           #  (2×1)

        # ───── Update step ──────────────────────────────────────────────────
        x_filt         = x_pred[t] + K.flatten() * innovations[t]
        P_filt         = (np.eye(2) - K @ C) @ P_pred[t]

        # accumulate (Gaussian) log-lik
        ll += -0.5 * (np.log(2*np.pi*S[t]) + innovations[t]**2 / S[t])

    return x_pred, P_pred, innovations, S, x_filt, P_filt, -ll                                 # NEGATIVE log-lik for minimise


def objective(params, y, U):
    _, _, _, _, _, _, L =kf_logLik_2d(params, y, U)
    return L


def estimate_2d(df, start_guess, bounds):
    y = df["Y"].values
    U = df[["Ta", "S", "I"]].values.T      # shape (3, n) for matrix algebra
    res = minimize(objective,
                   x0=start_guess, args=(y, U),
                   method="L-BFGS-B", bounds=bounds)
    return res

rng = np.random.default_rng(42)

A0  = rng.uniform(-0.5, 0.5, size=(2, 2))
B0  = rng.uniform(-1, 1,  size=(2, 3))
C0  = rng.uniform(-1, 1,  size=(1, 2))
Q0  = np.full(2,  1)          # small process noise
R0  = 1
X00 = rng.normal([20.0, 0.0], [1.0, 0.1])
P00 = np.full(2,  10.0)

start = np.concatenate([A0.ravel(), B0.ravel(), C0.ravel(),
                        Q0, [R0], X00, P00])

bounds =  [(-1, 1)]*4            # A
bounds += [(-2, 2)]*6            # B
bounds += [(-1, 1)]*2            # C
bounds += [(1e-8, None)]*2       # Q (diag)
bounds += [(1e-8, None)]         # R
bounds += [(0, 100)]*2           # X0
bounds += [(1e-8, 100)]*2        # P0 (diag)

df = raw
result = estimate_2d(raw, start, bounds)
print(result.success, result.fun, result.x)

# Filter once more with the MLEs so you can get smoothed states, CIs, etc.
y     = raw["Y"].values
U     = raw[["Ta", "S", "I"]].values.T
negLL = kf_logLik_2d(result.x, y, U)         # also gives you ŷ, P̂, ...

x_pred, P_pred, innovations, S, x_filt, P_filt, neglogLik = kf_logLik_2d(result.x,y,U)
print("done")

Y_pred = result.x[4]*x_pred
R = result.x[6]
C = result.x[4]
stderr = np.sqrt(C**2 * P_pred + R)
Y_pred_lower = Y_pred-1.96*stderr
Y_pred_higher  =  Y_pred+1.96*stderr

#Simple conf interval with true y
plt.plot(Y_pred_lower, label = "Conf interval", color = "black", linestyle='-',linewidth=0.5)   
plt.plot(Y_pred_higher, color = "black", linestyle='-',linewidth=0.5)
plt.plot(df["Y"], label="True $y_t$")
plt.plot(Y_pred, label="$\\hat{Y}_{t+1|t}$", linestyle='-.')
plt.legend()
plt.title("Kalman Filtering Results")
plt.savefig(f"{PARENT_DIR}/images/2.2/Conf_interval_1step.png")
plt.clf()

#Simple plot of innovation
plt.plot(innovations, label = "residuals")
plt.title("Residuals of 1 step predictions")
plt.savefig(f"{PARENT_DIR}/images/2.2/Residuals.png")
plt.clf()

#Simple plot of ACF PACF of residuals
lags = 40  # or however many lags you want
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# ACF
plot_acf(innovations, lags=lags, ax=axes[0])
axes[0].set_title("ACF of 1-step residuals")

# PACF
plot_pacf(innovations, lags=lags, ax=axes[1], method="ywm")
axes[1].set_title("PACF of 1-step residuals")

plt.tight_layout()
plt.savefig(f"{PARENT_DIR}/images/2.2/ACF_PACF.png")
# 1) QQ–plot of innovations
plt.figure()
sm.qqplot(innovations, line="45", fit=True)
plt.title("QQ-Plot of One‐Step Residuals")
plt.savefig(f"{PARENT_DIR}/images/2.2/QQplot_residuals.png")
plt.clf()

# 2) AIC and BIC
n = len(y)
k = len(result.x)
lnL = -neglogLik

AIC = 2*k - 2*lnL
BIC = k*np.log(n) - 2*lnL

print(f"AIC = {AIC:.2f}")
print(f"BIC = {BIC:.2f}")

# (Optionally save to file or DataFrame)
info = pd.DataFrame({
    "stat": ["n_obs","n_params","logLik","AIC","BIC"],
    "value": [n, k, lnL, AIC, BIC]
})
info.to_csv(f"{PARENT_DIR}/images/2.2/info_criteria.csv", index=False)


# ------------------------------------------------------------------
# Combined 2 × 2 chart
# ------------------------------------------------------------------
lags = 40                 # or whatever you prefer
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax_resid, ax_qq), (ax_acf, ax_pacf) = axes

# 1) One‑step residuals
ax_resid.plot(innovations, label="Residuals")
ax_resid.set_title("Residuals of 1‑Step Predictions")
ax_resid.legend()

# 2) QQ–plot of residuals
sm.qqplot(innovations, line="45", fit=True, ax=ax_qq)   # send it to the right axes
ax_qq.set_title("QQ-Plot of One-Step Residuals")

# 3) ACF of residuals
plot_acf(innovations, lags=lags, ax=ax_acf)
ax_acf.set_title("ACF of 1‑Step Residuals")

# 4) PACF of residuals
plot_pacf(innovations, lags=lags, ax=ax_pacf, method="ywm")
ax_pacf.set_title("PACF of 1‑Step Residuals")

plt.tight_layout()
plt.savefig(f"{PARENT_DIR}/images/2.2/diagnostic_2x2.png")
