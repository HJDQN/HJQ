from gym.envs.registration import register


register(
    id='LinearQuadraticRegulator1D-v0',
    entry_point='gym_lqr.envs:LinearQuadraticRegulator1DEnv',
    max_episode_steps=200,
)

register(
    id='LinearQuadraticRegulator20D-v0',
    entry_point='gym_lqr.envs:LinearQuadraticRegulator20DEnv',
    max_episode_steps=200,
)

register(
    id='LinearQuadraticRegulator30D-v0',
    entry_point='gym_lqr.envs:LinearQuadraticRegulator30DEnv',
    max_episode_steps=200,
)
register(
    id='LinearQuadraticRegulator50D-v0',
    entry_point='gym_lqr.envs:LinearQuadraticRegulator50DEnv',
    max_episode_steps=200,
)

