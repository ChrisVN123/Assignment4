import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from paths import PARENT_DIR

# Define x range
x = np.linspace(-5, 5, 400)

# Compute distributions
normal_pdf = norm.pdf(x)
t_dfs = [100, 5, 2, 1]
t_pdfs = {df: t.pdf(x, df) for df in t_dfs}

# Plot distributions
plt.figure()
plt.plot(x, normal_pdf, label="Normal Distribution")
for df in t_dfs:
    plt.plot(x, t_pdfs[df], label=f"t-distribution (df={df})")
plt.title("Normal Distribution vs. t-Distributions")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.savefig(f"{PARENT_DIR}/images/normal_vs_t_distributions.png", dpi=300)
plt.show()