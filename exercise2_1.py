import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("transformer_data.csv")
print(df.columns)
df.columns = ['time [hours]', 'Y (transformer temp.)[°C]', 'T (outdoor temp)[°C]', 'S (solar radiation)[W/m2]', 'I (transformer load)[kA]']

fig, ax1 = plt.subplots()

# Plot all series except 'S' and 'time' on the primary y-axis
#names = ['time', 'Y (transformer temp.)', 'T (outdoor temp)', 'S (solar radiation)', 'I (Transformer load)']
for name in df.columns:
    if name not in ["time [hours]", "S (solar radiation)[W/m2]"]:
        ax1.plot(df[name].values, label=name)

# Create secondary y-axis for 'S'
ax2 = ax1.twinx()
ax2.plot(df["S (solar radiation)[W/m2]"].values, color='black', label="S (solar radiation)[W/m2]", linestyle='--')

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

# Label axes
ax1.set_ylabel("Temperatur (°C)")
ax2.set_ylabel("W/m2")
ax1.set_xlabel("time (hours)")  # Add x-axis label

#plt.show()
