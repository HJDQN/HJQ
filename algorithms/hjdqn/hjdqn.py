import csv
import time
import gym
import mujoco_py
from algorithms.hjdqn.hjdqn_agent import HJDQNAgent
from algorithms.utils import set_log_dir, get_env_spec, scaled_env
from algorithms.noise import IndependentGaussian, Zero, SDE
import gym_lqr
from gym.envs.mujoco import MujocoEnv


def run_hjdqn(env_id,
              L=10.0,
              gamma=0.99,
              lr=1e-3,
              sigma=0.15,
              polyak=1e-3,
              hidden1=256,
              hidden2=256,
              max_iter=1e6,
              buffer_size=1e6,
              fill_buffer=20000,
              batch_size=128,
              train_interval=50,
              start_train=10000,
              eval_interval=2000,
              smooth=False,
              double=True,
              noise='gaussian',
              ep_len=None,
              h_scale=1.0,
              device='cpu',
              render=False,
              ):
    """
    param env_id: registered id of the environment
    param L: size of control constraint
    param gamma: discount factor, corresponds to 1 - gamma * h
    param lr: learning rate of optimizer
    param sigma: noise scale of Gaussian noise
    param polyak: target smoothing coefficient
    param hidden1: number of nodes of hidden layer1 of critic
    param hidden2: number of nodes of hidden layer2 of critic
    param max_iter: total number of environment interactions
    param buffer_size: size of replay buffer
    param fill_buffer: number of execution of random policy
    param batch_size: size of minibatch to be sampled during training
    param train_interval: length of interval between consecutive training
    param start_train: the beginning step of training
    param eval_interval: length of interval between evaluation
    param h_scale: scale of timestep of environment
    param device: device used for training
    param render: bool type variable for rendering
    """
    args = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)
    num_checkpoints = 5
    checkpoint_interval = max_iter // (num_checkpoints - 1)

    # create environment
    # adjust time step length, episode length if needed
    env = scaled_env(env_id=env_id, scale_factor=h_scale)
    test_env = scaled_env(env_id=env_id, scale_factor=h_scale)

    if ep_len is None:

        max_ep_len = env._max_episode_steps
    else:
        # in case episode limit is specified as a parameter
        max_ep_len = ep_len

    dimS, dimA, h, ctrl_range = get_env_spec(env)

    # scale gamma & learning rate
    gamma = 1. - h_scale * (1. - gamma)
    lr = h_scale * lr

    # create agent
    agent = HJDQNAgent(dimS, dimA, ctrl_range,
                       gamma,
                       h, L, sigma,
                       hidden1,
                       hidden2,
                       lr,
                       polyak,
                       buffer_size,
                       batch_size,
                       smooth=smooth,
                       device=device,
                       double=double,
                       render=render,
                       scale_factor=h_scale)

    # logger set-up

    set_log_dir(env_id)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/HJDQN_' + current_time + '.csv',
                     'w',
                     encoding='utf-8',
                     newline='')

    eval_log = open('./eval_log/' + env_id + '/HJDQN_' + current_time + '.csv',
                    'w',
                    encoding='utf-8',
                    newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)

    with open('./eval_log/' + env_id + '/HJDQN_' + current_time + '.txt', 'w') as f:
        for key, val in args.items():
            print(key, '=', val, file=f)

    # set noise process for exploration
    if noise == 'gaussian':
        noise_process = IndependentGaussian(dim=dimA, sigma=sigma)
    elif noise == 'sde':
        print('noise set to Ornstein-Uhlenbeck process')
        noise_process = SDE(dim=dimA, sigma=sigma, dt=h)
    else:
        print('unidentified noise type : noise process is set to zero')
        noise_process = Zero(dim=dimA)

    # start environment roll-out
    state = env.reset()
    noise = noise_process.reset()
    step_count = 0
    ep_reward = 0.

    action = env.action_space.sample()

    # main loop
    for t in range(max_iter + 1):

        # t : number of env-agent interactions (=number of transition samples observed)
        if t < fill_buffer:
            # first collect sufficient number of samples during the initial stage
            action = env.action_space.sample()
        else:
            action = agent.get_action(state, action, noise)
            noise = noise_process.sample()

        next_state, reward, done, _ = env.step(action)  # env-agent interaction
        step_count += 1

        if step_count == max_ep_len:
            # when the episode roll-out is truncated artificially(so that done=True), set done=False
            # thus, done=True only if the state is a terminal state
            done = False

        agent.buffer.append(state, action, reward, next_state, done)    # save the transition sample

        ep_reward += reward
        state = next_state

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])

            # restart an episode
            state = env.reset()
            noise = noise_process.reset()
            action = env.action_space.sample()
            step_count = 0
            ep_reward = 0.

        # Start training after sufficient number of transition samples are gathered
        if (t >= start_train) and (t % train_interval == 0):
            for _ in range(train_interval):
                agent.train()

        if t % eval_interval == 0:
            eval_data = agent.eval(test_env, t)
            eval_logger.writerow(eval_data)

        if t % checkpoint_interval == 0:
            if smooth:
                agent.save_model('./checkpoints/' + env_id + '/hjdqn_{}th_iter_smooth_'.format(t))
            else:
                agent.save_model('./checkpoints/' + env_id + '/hjdqn_{}th_iter_'.format(t))

    train_log.close()
    eval_log.close()

    return
