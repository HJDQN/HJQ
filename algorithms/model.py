import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    implementation of actor network mu
    2-layer mlp with tanh output layer
    """

    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, ctrl_range):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_dim)

        self.ctrl_range = nn.Parameter(torch.Tensor(ctrl_range), requires_grad=False)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.ctrl_range * x

        return x


class Critic(nn.Module):
    """
    implementation of critic network Q(s, a)
    2 layer mlp
    """
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
