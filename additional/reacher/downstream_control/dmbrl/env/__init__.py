"""Custom MuJoCo environments.

`MBRLCartpole-v0` uses Gymnasium (``gymnasium`` + ``mujoco``). Other envs still target
legacy OpenAI Gym until they are ported.
"""

try:
    from gymnasium.envs.registration import register as register_gymnasium

    register_gymnasium(
        id="MBRLCartpole-v0",
        entry_point="dmbrl.env.cartpole:CartpoleEnv",
    )
    register_gymnasium(
        id="MBRLReacher3D-v0",
        entry_point="dmbrl.env.reacher:Reacher3DEnv",
    )
except ImportError:
    pass

try:
    from gym.envs.registration import register as register_gym

    register_gym(
        id="MBRLPusher-v0",
        entry_point="dmbrl.env.pusher:PusherEnv",
    )
    register_gym(
        id="MBRLHalfCheetah-v0",
        entry_point="dmbrl.env.half_cheetah:HalfCheetahEnv",
    )
except ImportError:
    pass
