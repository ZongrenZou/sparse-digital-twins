import numpy as np
import matplotlib.pyplot as plt


ratios1 = np.load("./figs/ratios_1.npy")
ratios2 = np.load("./figs/ratios_100.npy")


fig, ax = plt.subplots(1, 1, figsize=[15, 5], dpi=500)
index = ["$x_{}$".format(str(i+1)) for i in range(100)]
index += ["$u_1$", "$u_2$"]
increments1 = ratios1[1:] - ratios1[:-1]
increments2 = ratios2[1:] - ratios2[:-1]

ax.plot(index, increments1, "k-o", linewidth=2, markersize=10)
ax.plot(index, increments2, "r--*", linewidth=2, markersize=10)
ax.set_xticks([])

ax.legend(["$\Delta x_1$", "$\Delta x_{100}$"], fontsize=60, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig("./figs/ratios.png")
