import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dim', required=True, type=int)

args = parser.parse_args()

n = args.dim

A = .1 * np.random.rand(n, n)
B = .5 * np.random.rand(n, n)

np.save('data/A{}.npy'.format(n), A)
np.save('data/B{}.npy'.format(n), B)

