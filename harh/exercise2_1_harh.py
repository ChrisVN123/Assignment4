import pandas as pd
from paths import PARENT_DIR
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

raw = pd.read_csv(f"{PARENT_DIR}/transformer_data.csv")
cols = raw.columns

fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(10, 3 * len(cols)))
color_list = list(mcolors.TABLEAU_COLORS.values())
labels = ["time (h)","transformer temp (C)","Outdoor temp (C)","Solar Radiation (W/m^2)","load (kA)"]
for i, (ax, col) in enumerate(zip(axes, cols)):
    color = color_list[i % len(color_list)]
    ax.plot(raw[col], label=labels[i], color=color)
    ax.set_title(col)
    ax.legend()

plt.tight_layout()
plt.savefig(f"{PARENT_DIR}/images/raw_data.png")
