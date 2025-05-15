import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2025)

def simulated(a, b, sigma1):
    #parameters
    sigma2 = 1  
    X0 = 5

    n = 100

    #initialize arrays
    X = np.zeros(n)
    Y = np.zeros(n)

    #initial values
    X[0] = X0
    Y[0] = X[0] + np.random.normal(0, sigma2)

    #simulate the process
    for t in range(1, n):
        if v is None:
            e1 = np.random.normal(0, sigma1)
        else:
            e1 = np.random.standard_t(v)
        X[t] = a * X[t-1] + b + e1

        e2 = np.random.normal(0, sigma2)
        Y[t] = X[t] + e2

    return X,Y


#plotting
if __name__ == "__main__":
    X,Y = simulated(a=1, b=0.9, sigma1=1)
    plt.figure(figsize=(10, 6))
    plt.plot(X, label="$X_t$ (true state)", linewidth=2)
    plt.plot(Y, label="$Y_t$ (observed)", linestyle='--')
    plt.title("Simulation of One Realization of the State-Space Model")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

