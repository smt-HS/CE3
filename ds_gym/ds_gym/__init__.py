from gym.envs.registration import register

register(
    id='ds-v102',
    entry_point='ds_gym.envs:DStSNEEnv',
)
