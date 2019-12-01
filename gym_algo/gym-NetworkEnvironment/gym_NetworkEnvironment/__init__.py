from gym.envs.registration import register

register(
    id='NetworkEnvironment-v0',
    entry_point='gym_NetworkEnvironment.envs:myNetworkEnvironment',
)