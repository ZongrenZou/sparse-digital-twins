import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


data = sio.loadmat("figs/surrogate_mpc_results.mat")

print(data.keys())


# case 1
x_traj = data["MPC_known_x_traj"]
x_traj_full = data["MPC_surrogate_unpruned_x_traj"]
x_traj_pruned = data["MPC_surrogate_pruned_x_traj"]
total_cost = data["MPC_known_cum_total_cost"]
total_cost_full = data["MPC_surrogate_unpruned_cum_total_cost"]
total_cost_pruned = data["MPC_surrogate_pruned_cum_total_cost"]




plt.figure(figsize=[6, 4])
x_traj = np.mean(x_traj, axis=0)
x_traj_full = np.mean(x_traj_full, axis=0)
x_traj_pruned = np.mean(x_traj_pruned, axis=0)
plt.plot(np.linspace(0, 1, 100), x_traj[0, :], "gray", linewidth=2, label="Initial condition")
plt.plot(np.linspace(0, 1, 100), x_traj[-1, :], "k-", linewidth=2, label="The exact model")
plt.plot(np.linspace(0, 1, 100), x_traj_full[-1, :], "r--", linewidth=2, markersize=10, label="The dense surrogate")
plt.plot(np.linspace(0, 1, 100), x_traj_pruned[-1, :], "m-.", linewidth=2, markersize=10, label="The sparse surrogate")
# plt.xlabel("x")
# plt.ylabel("Temperature")
# plt.title("End profile comparison")
plt.legend(frameon=False, fontsize=20)
plt.tight_layout()
plt.savefig("./figs/end_profile_comparison.png", dpi=300)
# plt.show()


total_cost = np.mean(total_cost, axis=0)
total_cost_full = np.mean(total_cost_full, axis=0)
total_cost_pruned = np.mean(total_cost_pruned, axis=0)

plt.figure(figsize=[6, 4])
plt.plot(total_cost, "k-", linewidth=2, label="The exact model")
plt.plot(total_cost_full, "r--", label="The dense surrogate")
plt.plot(total_cost_pruned, "m-.", linewidth=2, label="The sparse surrogate")
# plt.xlabel("Time step")
# plt.ylabel("Total cost")
# plt.title("Cumulative running cost (state + control) comparison")
plt.legend(frameon=False, fontsize=20)
plt.tight_layout()
plt.savefig("./figs/total_cost_comparison.png", dpi=300)
# plt.show()

print(total_cost_full[-1])
print(total_cost_pruned[-1])
print(total_cost[-1])

print("End main.")