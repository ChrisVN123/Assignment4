import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("transformer_data.csv")

fig, ax1 = plt.subplots()

# Plot all series except 'S' and 'time' on the primary y-axis
for name in df.columns:
    if name not in ["time", "S"]:
        ax1.plot(df[name].values, label=name)

# Create secondary y-axis for 'S'
ax2 = ax1.twinx()
ax2.plot(df["S"].values, color='black', label="S", linestyle='--')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

# Label axes
ax1.set_ylabel("Other Series")
ax2.set_ylabel("S Series")
ax1.set_xlabel("time")  # Add x-axis label

plt.show()
