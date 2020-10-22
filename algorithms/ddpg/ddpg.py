import time
import csv
import gym
import mujoco_py
import gym_lqr
from algorithms.ddpg.ddpg_agent import DDPGAgent
from algorithms.utils import get_env_spec, set_log_dir


def run_ddpg(env_id,
             gamma=0.99,
             actor_lr=1e-4,
             critic_lr=1e-3,
             polyak=1e-3,
             sigma=0.1,
             hidden_size1=256,
             hidden_size2=256,
             max_iter=1e6,
             eval_interval=2000,
             start_train=10000,
             train_interval=50,
             buffer_size=1e6,
             fill_buffer=20000,
             batch_size=128,
             h_scale=1.0,
             device='cpu',
             render='False'
             ):
    """
    :param env_id: registered id of the environment
    :param gamma: discount factor
    :param actor_lr: learning rate of actor optimizer
    :param critic_lr: learning rate of critic optimizer
    :param sigma: noise scale of Gaussian noise
    :param polyak: target smoothing coefficient
    :param hidden1: number of nodes of hidden layer1 of critic
    :param hidden2: number of nodes of hidden layer2 of critic
    :param max_iter: total number of environment interactions
    :param buffer_size: size of replay buffer
    :param fill_buffer: number of execution of random policy
    :param batch_size: size of minibatch to be sampled during training
    :param train_interval: length of interval between consecutive training
    :param start_train: the beginning step of training
    :param eval_interval: length of interval between evaluation
    :param h_scale: scale of timestep of environment
    :param device: device used for training
    :param render: bool type variable for rendering
    """
    args = locals()

    max_iter = int(max_iter)
    buffer_size = int(buffer_size)

    num_checkpoints = 5
    checkpoint_interval = max_iter // (num_checkpoints - 1)

    env = gym.make(env_id)

    dimS, dimA, _, ctrl_range, max_ep_len = get_env_spec(env)

    agent = DDPGAgent(dimS,
                      dimA,
                      ctrl_range,
                      gamma=gamma,
                      actor_lr=actor_lr,
                      critic_lr=critic_lr,
                      polyak=polyak,
                      sigma=sigma,
                      hidden_size1=hidden_size1,
                      hidden_size2=hidden_size2,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      h_scale=h_scale,
                      device=device,
                      render=render)

    set_log_dir(env_id)
    current_time = time.strftime("%m%d-%H%M%S")
    train_log = open('./train_log/' + env_id + '/DDPG_' + current_time + '.csv',
                     'w',
                     encoding='utf-8',
                     newline='')
    eval_log = open('./eval_log/' + env_id + '/DDPG_' + current_time + '.csv',
                    'w',
                    encoding='utf-8',
                    newline='')

    train_logger = csv.writer(train_log)
    eval_logger = csv.writer(eval_log)
    with open('./eval_log/' + env_id + '/DDPG_' + current_time + '.txt', 'w') as f:
        for key, val in args.items():
            print(key, '=', val, file=f)

    state = env.reset()
    step_count = 0
    ep_reward = 0

    # main loop
    for t in range(max_iter + 1):
        if t < fill_buffer:
            # first collect sufficient number of samples during the initial stage
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)  # env-agent interaction
        step_count += 1

        if step_count == max_ep_len:
            done = False

        agent.buffer.append(state, action, reward, next_state, done)    # save the transition sample

        state = next_state
        ep_reward += reward

        if done or (step_count == max_ep_len):
            train_logger.writerow([t, ep_reward])
            state = env.reset()
            step_count = 0
            ep_reward = 0

        if (t >= start_train) and (t % train_interval == 0):
            # Start training after sufficient number of transition samples are gathered
            for _ in range(train_interval):
                agent.train()

        if t % eval_interval == 0:
            log = agent.eval(env_id, t)
            eval_logger.writerow(log)

        if t % checkpoint_interval == 0:
            agent.save_model('./checkpoints/' + env_id + '/ddpg_{}th_iter_'.format(t))

    train_log.close()
    eval_log.close()

    return

