import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.io as sio
import numpy as np
import diffrax as dfx


from ckpt_io import save_model
import models


def sample(
    params,
    x,
    model,
):  
    # define velocity function
    def velocity(params, model, t, z, x):
        out = model.apply({"params": params}, t, z, x)
        return out.reshape(-1)
        
    # Build once
    term = dfx.ODETerm(lambda t, y, args: velocity(params, model, t, y, args))

    @jax.jit
    def solve_single(z0i, x):
        sol = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            t0=0.0,
            t1=1.0,
            dt0=None,
            y0=z0i,
            args=x,  # pass conditioning vector for this traj
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=1_000_000,
        )
        return sol.ys

    z0 = np.random.normal(size=[x.shape[0], 1])
    z0 = jnp.array(z0)
    x_fixed = x
    zT = jax.vmap(solve_single, in_axes=(0, 0))(z0, x_fixed)
    return zT, z0



if __name__ == "__main__":
    M = 95_000

    data = sio.loadmat("./data/pendulum_data.mat")
    theta = data["theta"].reshape([500, 200])
    theta_next = data["theta_next"].reshape([500, 200])
    u = data["u"].reshape([500, 200])
    y_data = theta_next - theta

    x_data = np.stack(
        [
            np.sin(theta[:, 9:]),
            np.sin(theta[:, 8:-1]),
            np.sin(theta[:, 7:-2]),
            np.sin(theta[:, 6:-3]),
            np.sin(theta[:, 5:-4]),
            np.sin(theta[:, 4:-5]),
            np.sin(theta[:, 3:-6]),
            np.sin(theta[:, 2:-7]),
            np.sin(theta[:, 1:-8]),
            np.sin(theta[:, :-9]),
            np.cos(theta[:, 9:]),
            np.cos(theta[:, 8:-1]),
            np.cos(theta[:, 7:-2]),
            np.cos(theta[:, 6:-3]),
            np.cos(theta[:, 5:-4]),
            np.cos(theta[:, 4:-5]),
            np.cos(theta[:, 3:-6]),
            np.cos(theta[:, 2:-7]),
            np.cos(theta[:, 1:-8]),
            np.cos(theta[:, :-9]),
            u[:, 9:],
            u[:, 8:-1],
            u[:, 7:-2],
            u[:, 6:-3],
            u[:, 5:-4],
            u[:, 4:-5],
            u[:, 3:-6],
            u[:, 2:-7],
            u[:, 1:-8],
            u[:, :-9],
        ],
        axis=-1,
    )
    y_data = y_data[:, 9:]

    x_data = x_data.reshape([-1, 30])[:M]
    y_data = y_data.reshape([-1, 1])[:M]

    seed = 81763263  # for reproducibility
    np.random.seed(seed)

    ## normalize the data
    x_mu = np.mean(x_data, axis=0)
    x_sd = np.std(x_data, axis=0)
    y_mu = np.mean(y_data, axis=0)
    y_sd = np.std(y_data, axis=0)
    X_train = (x_data - x_mu) / x_sd
    Y_train = (y_data - y_mu) / y_sd

    key = np.random.randint(0, 1_000_000_000)
    key = jr.PRNGKey(key)

    ############ Training ############
    params, model = models.train_cfm_flax(
        key,
        X_train,
        Y_train,
        model_cfg=models.ModelCfg(
            hidden_dims=[64, 64, 64, 64],
        ),
        train_cfg=models.TrainCfg(
            lr=1e-4,
            batch_size=1_000,
            steps=1_000,
        ),
    )

    save_model(
        ckpt_dir="./checkpoints/cfm_all",
        params=params,
        model=model,
        model_cfg=models.ModelCfg(hidden_dims=[64, 64, 64, 64]),
        extras={
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    ############ Sampling ############
    x_data = x_data.reshape([-1, 30])
    M = x_data.shape[0]
    N = 10
    x = np.tile(x_data, [N, 1]).reshape([N * M, x_data.shape[1]])

    z_samples, z0_samples = sample(
        params,
        x=(x - x_mu) / x_sd,
        model=model,
    )
    y_samples = z_samples[:, 0, :] * y_sd + y_mu

    sio.savemat(
        "./outputs/samples.mat",
        {
            "y_samples": y_samples,
            "z_samples": z0_samples,
            "x_samples": x,
            "x_mu": x_mu,
            "x_sd": x_sd,
            "y_mu": y_mu,
            "y_sd": y_sd,
        },
    )

    print("End main.")
