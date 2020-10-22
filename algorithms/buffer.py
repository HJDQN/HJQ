import numpy as np
from collections import namedtuple


class ReplayBuffer:
    """
    implementation of a replay buffer to store transition samples
    """
    def __init__(self, state_dim, action_dim, limit):
        self.states = CircularBuffer(shape=(state_dim,), limit=limit)
        self.actions = CircularBuffer(shape=(action_dim,), limit=limit)
        self.rewards = CircularBuffer(shape=(1,), limit=limit)
        self.next_states = CircularBuffer(shape=(state_dim,), limit=limit)
        self.terminals = CircularBuffer(shape=(1,), limit=limit)

        self.limit = limit  # maximum capacity of the buffer
        self.size = 0       # current size of the buffer

    def append(self, state, action, reward, next_state, done):
        # store a given transition sample to the buffer
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)

        self.size = self.states.size

    def sample_batch(self, batch_size):
        # uniformly sample transition data from the buffer
        rng = np.random.default_rng()
        idxs = rng.choice(self.size, batch_size)    # sample indices

        # get batch from each buffer
        state_batch = self.states.get_batch(idxs)
        action_batch = self.actions.get_batch(idxs)
        reward_batch = self.rewards.get_batch(idxs)
        next_state_batch = self.next_states.get_batch(idxs)
        terminal_batch = self.terminals.get_batch(idxs)

        batch = Batch(state=state_batch,
                      act=action_batch,
                      next_state=next_state_batch,
                      rew=reward_batch,
                      done=terminal_batch)

        return batch


Batch = namedtuple('Batch', ['state', 'act', 'next_state', 'rew', 'done'])      # batch definition as python named tuple


class CircularBuffer:
    """
    implementation of a circular buffer
    based on OpenAI baseline implementation of replay buffer:
    https://github.com/openai/baselines/blob/master/baselines/ddpg/memory.py
    """
    def __init__(self, shape, limit=1000000):
        self.start = 0
        self.data_shape = shape
        self.size = 0
        self.limit = limit
        self.data = np.zeros((self.limit,) + shape)

    def append(self, data):
        if self.size < self.limit:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.limit

        self.data[(self.start + self.size - 1) % self.limit] = data

    def get_batch(self, idxs):

        return self.data[(self.start + idxs) % self.limit]
