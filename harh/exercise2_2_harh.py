from paths import PARENT_DIR
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

raw = pd.read_csv(f"{PARENT_DIR}/transformer_data.csv")

def unpack_params(params):
    A  = params[0]
    B  = params[1:4]        # length-3 vector
    Q  = params[4]
    R  = params[5]
    X0 = params[6]
    P0 = params[7]
    return A,B,Q,R,X0,P0

def kf_logLik_dt(params,y,U,C):
    #Some adjustable params
        # unpack
    A,B,Q,R,X0,P0 = unpack_params(params)

    # Allocate arrays
    n = len(y)
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
        x_pred[t] = A * x_filt[t-1] + B.dot(U[t-1]) 
        P_pred[t] = A * P_filt[t-1] * A.transpose() + Q

        # Innovation
        innovations[t] = y[t] - C*x_pred[t]
        S[t] = C*P_pred[t]*C.transpose() + R

        # Kalman Gain
        K_t = P_pred[t] * C.transpose() / S[t]

        # Update
        x_filt[t] = x_pred[t] + K_t * innovations[t]
        P_filt[t] = ((1 - K_t*C) * P_pred[t]).item()
        L += -1/2*np.sum(np.log(2*np.pi*S[t])+innovations[t]**2*(S[t]**(-1)))

    return x_pred, P_pred, innovations, S, x_filt, P_filt, -L
def objective(params,y,U,C):
    _, _, _, _, _, _, L_val = kf_logLik_dt(params,y,U,C)
    return L_val

def estimate_dt(df,start_guess,bounds):
    y = df["Y"].values
    U = df[["Ta","S","I"]].values
    C = np.array([1])
    result = minimize(
        fun=lambda p: objective(p, y, U,C),
        x0=start_guess,
        method="L-BFGS-B",
        bounds=bounds
    )
    return result

bounds = [
    (-1,1),  # A
    (-1,1),  # B1
    (-1,1),  # B2
    (-1,1),  # B3
    (1e-8,None),  # Q (variance ≥ 0)
    (1e-8,None),  # R
    (0,100),      # X0
    (1e-8,100)    # P0
]



A = 1
B = np.array([1,1,1])
Q = 1
R = 1
X0 = 23.5
P0 = 1
df = raw
start = np.concatenate([
    [A],       # 0
    B,         # 1–3
    [Q, R,  # 4–6
     X0, P0]   # 7–8
])
result = estimate_dt(df,start,bounds)
theta_hat = result.x 
A,B,Q,R,X0,P0 = unpack_params(theta_hat)

y = df["Y"].values
U = df[["Ta","S","I"]].values
C = np.array([1])
x_pred, P_pred, innovations, S, x_filt, P_filt, neglogLik = kf_logLik_dt(result.x,y,U,C)
print("done")

Y_pred = C*x_pred[1:]
R = result.x[6]
residuals = y[1:] - Y_pred
stderr = np.sqrt(C**2 * P_pred[1:] + R)
Y_pred_lower = Y_pred-1.96*stderr
Y_pred_higher  =  Y_pred+1.96*stderr

#Simple conf interval with true y
plt.plot(Y_pred_lower, label = "Conf interval", color = "black", linestyle='-',linewidth=0.5)   
plt.plot(Y_pred_higher, color = "black", linestyle='-',linewidth=0.5)
plt.plot(df["Y"][1:], label="True $y_t$")
plt.plot(Y_pred, label="$\\hat{Y}_{t+1|t}$", linestyle='-.')
plt.legend()
plt.title("Kalman Filtering Results (1d)")
plt.savefig(f"{PARENT_DIR}/images/2.2/ex2_2_confidence_int_prediction.png")
plt.clf()

#Simple plot of innovation
plt.plot(innovations[1:], label = "residuals")
plt.title("Residuals of 1 step predictions")
plt.savefig(f"{PARENT_DIR}/images/2.2/Residuals.png")
plt.clf()

#Simple plot of ACF PACF of residuals
lags = 40  # or however many lags you want
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

# ACF
plot_acf(innovations[1:], lags=lags, ax=axes[0])
axes[0].set_title("ACF of 1-step residuals")

# PACF
plot_pacf(innovations[1:], lags=lags, ax=axes[1], method="ywm")
axes[1].set_title("PACF of 1-step residuals")

plt.tight_layout()
plt.savefig(f"{PARENT_DIR}/images/2.2/ACF_PACF.png")
# 1) QQ–plot of innovations
plt.figure()
sm.qqplot(innovations[1:], line="45", fit=True)
plt.title("QQ-Plot of One‐Step Residuals")
plt.savefig(f"{PARENT_DIR}/images/2.2/QQplot_residuals.png")
plt.clf()

# 2) AIC and BIC
n = len(y)
k = len(result.x)
lnL = -neglogLik

AIC = 2*k - 2*lnL
BIC = k*np.log(n) - 2*lnL
RMSE = np.sqrt(np.mean(residuals**2))
print(f"AIC = {AIC:.2f}")
print(f"BIC = {BIC:.2f}")
print(f"RMSE = {RMSE:.2f}")


# (Optionally save to file or DataFrame)
info = pd.DataFrame({
    "stat": ["n_obs","n_params","logLik","AIC","BIC","RMSE"],
    "value": [n, k, lnL, AIC, BIC,RMSE]
})
info.to_csv(f"{PARENT_DIR}/images/2.2/info_criteria.csv", index=False)

####
unpack_params(result.x)
A,B,C,Q,R,X0
params = pd.DataFrame({
    "params": ["A", "B", "C", "Q", "R", "x0",],
    "value": [A, B, C, Q, R, X0]
})
params.to_csv(f"images/2.2/params.csv", index=False)
# ------------------------------------------------------------------
# Combined 2 × 2 chart
# ------------------------------------------------------------------
lags = 40                 # or whatever you prefer
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
(ax_resid, ax_qq), (ax_acf, ax_pacf) = axes

# 1) One‑step residuals
ax_resid.plot(innovations[1:], label="Residuals")
ax_resid.set_title("Residuals of 1‑Step Predictions")
ax_resid.legend()

# 2) QQ–plot of residuals
sm.qqplot(innovations[1:], line="45", fit=True, ax=ax_qq)   # send it to the right axes
ax_qq.set_title("QQ-Plot of One-Step Residuals")

# 3) ACF of residuals
plot_acf(innovations[1:], lags=lags, ax=ax_acf)
ax_acf.set_title("ACF of 1‑Step Residuals")

# 4) PACF of residuals
plot_pacf(innovations[1:], lags=lags, ax=ax_pacf, method="ywm")
ax_pacf.set_title("PACF of 1‑Step Residuals")

plt.tight_layout()
plt.savefig(f"{PARENT_DIR}/images/2.2/ex2_2_stats.png")
