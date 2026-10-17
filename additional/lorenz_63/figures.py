import numpy as np
import matplotlib.pyplot as plt

ratios1 = []
ratios2 = []
ratios3 = []
for i in range(10):
    ratios1.append(np.loadtxt("./outputs/ratios_1_{}.txt".format(str(i))))
    ratios2.append(np.loadtxt("./outputs/ratios_2_{}.txt".format(str(i))))
    ratios3.append(np.loadtxt("./outputs/ratios_3_{}.txt".format(str(i))))

ratios1 = np.stack(ratios1)
ratios2 = np.stack(ratios2)
ratios3 = np.stack(ratios3)

ratios1_mu = np.mean(ratios1, axis=0)
ratios2_mu = np.mean(ratios2, axis=0)
ratios3_mu = np.mean(ratios3, axis=0)
ratios1_sd = np.std(ratios1, axis=0)
ratios2_sd = np.std(ratios2, axis=0)
ratios3_sd = np.std(ratios3, axis=0)

fig, ax = plt.subplots(1, 2, figsize=[30, 5], dpi=500)

ax[0].errorbar(
    ["3", "2", "1", "0"],
    ratios1_mu,
    yerr=ratios1_sd,
    fmt="o",
    color="k",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[0].errorbar(
    ["3", "2", "1", "0"],
    ratios2_mu,
    yerr=ratios2_sd,
    fmt="s",
    color="b",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[0].errorbar(
    ["3", "2", "1", "0"],
    ratios3_mu,
    yerr=ratios3_sd,
    fmt="x",
    color="r",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)


increments1 = ratios1[:, 1:] - ratios1[:, :-1]
increments2 = ratios2[:, 1:] - ratios2[:, :-1]
increments3 = ratios3[:, 1:] - ratios3[:, :-1]

incre1_mu = np.mean(increments1, axis=0)
incre2_mu = np.mean(increments2, axis=0)
incre3_mu = np.mean(increments3, axis=0)
incre1_sd = np.std(increments1, axis=0)
incre2_sd = np.std(increments2, axis=0)
incre3_sd = np.std(increments3, axis=0)

ax[1].errorbar(
    ["1", "2", "3"],
    incre1_mu,
    yerr=incre1_sd,
    fmt="o",
    color="k",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[1].errorbar(
    ["1", "2", "3"],
    incre2_mu,
    yerr=incre2_sd,
    fmt="s",
    color="b",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[1].errorbar(
    ["1", "2", "3"],
    incre3_mu,
    yerr=incre3_sd,
    fmt="x",
    color="r",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
# ax[0].set_ylim([0, 1])
ax[1].set_ylim([0, 1])

ax[0].set_xticks([])
ax[1].set_xticks([])

ax[0].legend(
    ["$\Delta x_1$", "$\Delta x_2$", "$\Delta x_3$"],
    fontsize=40,
    frameon=False,
    loc="upper left",
)
# ax[1].set_legend(["$R_1$", "$R_2$", "$R_3$"])
# plt.fill_between(np.arange(len(ratios1_mu)), ratios1_mu - ratios1_sd, ratios1_mu + ratios1_sd, color="k", alpha=0.2)
# plt.fill_between(np.arange(len(ratios2_mu)), ratios2_mu - ratios2_sd, ratios2_mu + ratios2_sd, color="r", alpha=0.2)
# plt.fill_between(np.arange(len(ratios3_mu)), ratios3_mu - ratios3_sd, ratios3_mu + ratios3_sd, color="b", alpha=0.2)
plt.tight_layout()
plt.savefig("./ratios.png")


ratios1 = []
ratios2 = []
ratios3 = []
for i in range(10):
    ratios1.append(np.loadtxt("./outputs/ratios_1_diffusion_{}.txt".format(str(i))))
    ratios2.append(np.loadtxt("./outputs/ratios_2_diffusion_{}.txt".format(str(i))))
    ratios3.append(np.loadtxt("./outputs/ratios_3_diffusion_{}.txt".format(str(i))))

ratios1 = np.stack(ratios1)
ratios2 = np.stack(ratios2)
ratios3 = np.stack(ratios3)

ratios1_mu = np.mean(ratios1, axis=0)
ratios2_mu = np.mean(ratios2, axis=0)
ratios3_mu = np.mean(ratios3, axis=0)
ratios1_sd = np.std(ratios1, axis=0)
ratios2_sd = np.std(ratios2, axis=0)
ratios3_sd = np.std(ratios3, axis=0)

fig, ax = plt.subplots(1, 2, figsize=[30, 5], dpi=500)

ax[0].errorbar(
    ["3", "2", "1", "0"],
    ratios1_mu,
    yerr=ratios1_sd,
    fmt="o",
    color="k",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[0].errorbar(
    ["3", "2", "1", "0"],
    ratios2_mu,
    yerr=ratios2_sd,
    fmt="s",
    color="b",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[0].errorbar(
    ["3", "2", "1", "0"],
    ratios3_mu,
    yerr=ratios3_sd,
    fmt="x",
    color="r",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)


increments1 = ratios1[:, 1:] - ratios1[:, :-1]
increments2 = ratios2[:, 1:] - ratios2[:, :-1]
increments3 = ratios3[:, 1:] - ratios3[:, :-1]

incre1_mu = np.mean(increments1, axis=0)
incre2_mu = np.mean(increments2, axis=0)
incre3_mu = np.mean(increments3, axis=0)
incre1_sd = np.std(increments1, axis=0)
incre2_sd = np.std(increments2, axis=0)
incre3_sd = np.std(increments3, axis=0)

ax[1].errorbar(
    ["1", "2", "3"],
    incre1_mu,
    yerr=incre1_sd,
    fmt="o",
    color="k",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[1].errorbar(
    ["1", "2", "3"],
    incre2_mu,
    yerr=incre2_sd,
    fmt="s",
    color="b",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)
ax[1].errorbar(
    ["1", "2", "3"],
    incre3_mu,
    yerr=incre3_sd,
    fmt="x",
    color="r",
    linewidth=3,
    markersize=10,
    capsize=5,
    linestyle="-",
)

ax[0].set_xticks([])
ax[1].set_xticks([])
# ax[0].set_ylim([0, 1])
ax[1].set_ylim([0, 1])

ax[0].legend(["$R_1$", "$R_2$", "$R_3$"], fontsize=40, frameon=False, loc="upper left")
# ax[1].set_legend(["$R_1$", "$R_2$", "$R_3$"])
# plt.fill_between(np.arange(len(ratios1_mu)), ratios1_mu - ratios1_sd, ratios1_mu + ratios1_sd, color="k", alpha=0.2)
# plt.fill_between(np.arange(len(ratios2_mu)), ratios2_mu - ratios2_sd, ratios2_mu + ratios2_sd, color="r", alpha=0.2)
# plt.fill_between(np.arange(len(ratios3_mu)), ratios3_mu - ratios3_sd, ratios3_mu + ratios3_sd, color="b", alpha=0.2)
plt.tight_layout()
plt.savefig("./ratios_case_2.png")
