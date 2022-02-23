import torch
from torch import nn

# [Goal,State] -> num_actions
class DenseNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_channels):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        # (Batch, Number Channels, height, width)
        self.layer1 = nn.Sequential(
            nn.Linear(num_states * num_channels, num_states * 4),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(num_states * 4, num_states * 4),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(num_states * 4, num_states * 4),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(num_states * 4, num_states * 4),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(num_states * 4, num_states),
            nn.ReLU())
        self.layer6 = nn.Linear(num_states, num_actions)

    def forward(self, state):
        h = state
        h = torch.flatten(h,start_dim=1) # start_dim to maintain batch size
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        q_values = h
        return q_values

# [Goal,State] -> num_actions
class SmallDenseNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_channels):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        # (Batch, Number Channels, height, width)
        self.layer1 = nn.Sequential(
            nn.Linear(num_states * num_channels, num_states*4),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(num_states * 4, num_states*4),
            nn.ReLU())
        self.layer3 = nn.Linear(num_states*4, num_actions)

    def forward(self, state):
        h = state
        h = torch.flatten(h,start_dim=1) # start_dim to maintain batch size
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        q_values = h
        return q_values