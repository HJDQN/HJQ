import os
import gym
from gym.envs.mujoco import MujocoEnv

def freeze(net):
    for p in net.parameters():
        p.requires_grad_(False)


def unfreeze(net):
    for p in net.parameters():
        p.requires_grad_(True)


def get_env_spec(env):
    # load environment information from created environment
    print('environment : ' + env.unwrapped.spec.id)
    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    ctrl_range = env.action_space.high
    max_ep_len = env._max_episode_steps
    print('-' * 80)
    print('observation dim : {} / action dim : {}'.format(dimS, dimA))
    print('dt : {}'.format(env.dt))
    print('control range : {}'.format(ctrl_range))
    print('max_ep_len : ', max_ep_len)
    print('-' * 80)

    return dimS, dimA, env.dt, ctrl_range, max_ep_len


def set_log_dir(env_id):
    # set up directories to save logs
    if not os.path.exists('./train_log/'):
        os.mkdir('./train_log/')
    if not os.path.exists('./eval_log/'):
        os.mkdir('./eval_log/')

    if not os.path.exists('./train_log/' + env_id + '/'):
        os.mkdir('./train_log/' + env_id + '/')
    if not os.path.exists('./eval_log/' + env_id + '/'):
        os.mkdir('./eval_log/' + env_id + '/')

    if not os.path.exists('./checkpoints/'):
        os.mkdir('./checkpoints/')

    if not os.path.exists('./checkpoints/' + env_id + '/'):
        os.mkdir('./checkpoints/' + env_id + '/')

    return


def scaled_env(env_id, scale_factor):
    """
    adjust environment parameters related to time discretization
    """

    env = gym.make(env_id)

    if isinstance(env.unwrapped, MujocoEnv):
        ########################################################################################
        # time discretization in MuJoCo                                                        #
        # dt = env.model.opt.timestep * env.frame_skip                                         #
        # if scale factor = k, then dt <- k * dt, which is achieved by  frame_skip by k        #
        # physical time horizon T = episode steps * dt                                         #
        # episode length is also rescaled so that T remains the same                           #
        ########################################################################################
        if (env.frame_skip * scale_factor) != int(env.frame_skip * scale_factor):
            raise ValueError('invalid scale factor -> frame skip = {}, scale_factor =  {}'.format(env.frame_skip,
                                                                                                  scale_factor))

        env.unwrapped.frame_skip = int(env.frame_skip * scale_factor)

    else:
        env.dt *= scale_factor

    env._max_episode_steps = int(env._max_episode_steps / scale_factor)

    return env
