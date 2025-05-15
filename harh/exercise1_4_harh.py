import numpy as np
import matplotlib.pyplot as plt
from exercise1_2 import simulated
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
P0 = 10
theta = [1, 0.9, 1]  # a, b, Q
R = 1  # observation noise variance
# Apply Kalman filter to observations Y_t

def objective(th):
    _, _, _, _, _, _, L_val = myKalmanFilter(Y, th, R, X0, P0)
    return L_val



# lower, upper for each parameter
bounds = [
    (None, None),     # 1st variable:  –∞  … +∞
    (None, None),     # 2nd variable:  –∞  … +∞
    (1e-5, None)      # 3rd variable:   1 × 10⁻⁵ … +∞
]
params_container = np.zeros([100,3])
params1 = [1,0.9,1]
params2 = [5,0.9,1]
params3 = [1,0.9,5]
params = [params1,params2,params3]
for i,param in enumerate(params):
    for j in range(100):
        X, Y = simulated(param[0],param[1],param[2])
        result = minimize(objective, x0=param, method='L-BFGS-B', bounds=bounds)
        params_container[j] = result.x

        print("Optimized parameters:", result.x)
        print("Log-likelihood:", result.fun)
    # Plotting the results
    fig, ax = plt.subplots()
    bp = ax.boxplot(params_container,labels=[f"a={param[0]}",f"b={param[1]}",f"Sd1={param[2]}"], patch_artist=True)
    ax.scatter([1,2,3],
               param,
               color='red',
               marker='o',
               s=50,
               zorder=10,
               label='true value')
    plt.title(f"Boxplot of Optimized Parameters")
    print(f"Optimized parameters for initial values {param}: {result.x}")

plt.show()