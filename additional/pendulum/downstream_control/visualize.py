import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

returns = sio.loadmat("./logs/returns.mat")
ret_true = returns["ret_true"]
ret_fm5 = returns["ret_fm5"]
ret_v2_1 = returns["ret_v2_1"]
ret_v2_2 = returns["ret_v2_2"]
ret_w1 = returns["ret_w1"]

xb = np.arange(5)

means = [
    ret_true.mean(),
    ret_fm5.mean(),
    ret_v2_1.mean(),
    ret_v2_2.mean(),
    ret_w1.mean(),
]
stds = [ret_true.std(), ret_fm5.std(), ret_v2_1.std(), ret_v2_2.std(), ret_w1.std()]
colors = ["magenta", "steelblue", "red", "green", "orange"]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.set_xticks(xb)
ax.set_xticklabels(
    [
        "True \nenvironment",
        "Using all 10 \npast states",
        "Using discovered \ngraph 1",
        "Using discovered \ngraph 2",
        "Using only \nthe current state",
    ],
    fontsize=12,
)

ax.set_title("Return (sums of rewards) across 100 episodes")
ax.grid(True, axis="y", alpha=0.3)
ax.bar(xb, means, yerr=stds, capsize=5, color=colors, alpha=0.9, width=0.55)
for i, (mean, std, color) in enumerate(zip(means, stds, colors)):
    ax.text(
        xb[i],
        mean + std * 0.05 + 2,
        f"{mean:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        color="k",
    )
plt.tight_layout()
plt.savefig("./logs/returns.png", dpi=200)
plt.close()


trajs = sio.loadmat("./logs/trajs.mat")
trajs_true = trajs["trajs_true"][0:1, :]
trajs_fm5 = trajs["trajs_fm5"][0:1, :]
trajs_v2_1 = trajs["trajs_v2_1"][0:1, :]
trajs_v2_2 = trajs["trajs_v2_2"][0:1, :]
trajs_w1 = trajs["trajs_w1"][0:1, :]

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
dt = 0.05
t = dt * np.arange(trajs_true.shape[1]).reshape([-1, 1])
# theta_true = trajs_true.T
# theta_fm5 = trajs_fm5.T
# theta_v2_1 = trajs_v2_1.T
# theta_v2_2 = trajs_v2_2.T
# theta_w1 = trajs_w1.T
theta_true = np.arctan2(np.sin(trajs_true), np.cos(trajs_true)).T
theta_fm5 = np.arctan2(np.sin(trajs_fm5), np.cos(trajs_fm5)).T
theta_v2_1 = np.arctan2(np.sin(trajs_v2_1), np.cos(trajs_v2_1)).T
theta_v2_2 = np.arctan2(np.sin(trajs_v2_2), np.cos(trajs_v2_2)).T
theta_w1 = np.arctan2(np.sin(trajs_w1), np.cos(trajs_w1)).T

ax.plot(t, theta_true, "-", label="True physics", color="magenta")
ax.plot(t, theta_fm5, "-", label="Using all 10 past states", color="steelblue")
ax.plot(t, theta_v2_1, "-", label="Using discovered graph 1", color="red")
ax.plot(t, theta_v2_2, "-", label="Using discovered graph 2", color="green")
ax.plot(t, theta_w1, "-", label="Using only the current state", color="orange")

ax.legend(fontsize=12)
ax.set_xlabel("Time (s)", fontsize=12)
# ax.set_ylabel("Angle (rad)", fontsize=12)
ax.set_title("Angle trajectory of one episode", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./logs/trajs.png", dpi=200)
plt.close()


print("End main.")
