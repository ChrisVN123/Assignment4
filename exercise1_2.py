import numpy as np
import matplotlib.pyplot as plt

def simulated():
    #parameters
    a = 0.9
    b = 1
    sigma1 = 1  
    sigma2 = 1  
    X0 = 5
    n = 100

    np.random.seed(42)

    #initialize arrays
    X = np.zeros(n)
    Y = np.zeros(n)

    #initial values
    X[0] = X0
    Y[0] = X[0] + np.random.normal(0, sigma2)

    #simulate the process
    for t in range(1, n):
        e1 = np.random.normal(0, sigma1)
        X[t] = a * X[t-1] + b + e1

        e2 = np.random.normal(0, sigma2)
        Y[t] = X[t] + e2

    return X,Y

X,Y = simulated()
#plotting
if __name__ == "__main__":
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

