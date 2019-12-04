from gym.envs.registration import register

register(
    id='NetworkEnvironment-v1',
    entry_point='gym_NetworkEnvironment.envs:myNetworkEnvironment',
)