import numpy as np
from os import path
import argparse
import gym
import control
import gym_lqr


parser = argparse.ArgumentParser()
parser.add_argument('--dim', required=True, type=int)

args = parser.parse_args()

n = args.dim

A = np.load(path.join(path.dirname(__file__), "data/A{}.npy".format(n)))
B = np.load(path.join(path.dirname(__file__), "data/B{}.npy".format(n)))

m = B.shape[1]

Q = 5. * np.eye(n)
R = np.eye(m)

env = gym.make("LinearQuadraticRegulator{}D-v0".format(n))

ep_len = env._max_episode_steps
h = env.dt
gamma = .99999

conti_gamma = (1 - gamma) / h

# need to solve discounted CARE
A1 = A - (conti_gamma / 2.) * np.eye(n)

K, S, E = control.lqr(A1, B, Q, R)
K = np.asarray(K)


num_seeds = 100
score_arr = np.zeros(num_seeds)


for i in range(num_seeds):
    x = env.reset() 
    score =.0

    for t in range(ep_len):
        u = -K @ x
        x, reward, done, _ = env.step(u)
        score_arr[i] += gamma**t * reward


print('mean score : ', np.mean(score_arr))
np.save('data/optimal_d{}.npy'.format(n), score_arr)
