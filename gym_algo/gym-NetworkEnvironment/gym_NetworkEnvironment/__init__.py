from gym.envs.registration import register

register(
    id='NetworkEnvironment-v2',
    entry_point='gym_NetworkEnvironment.envs:myNetworkEnvironment',
)