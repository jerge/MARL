import torch
from torch import nn

# [Goal,State] -> num_actions
class ConvNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_channels):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        # (Batch, Number Channels, height, width)
        keep_prob = .9
        out_channels = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(self._num_channels,out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'), #num_states[0], num_states[1]
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            #nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=1 - keep_prob))
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        self.layer3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            # nn.MaxPool2d(kernel_size=2, out_channels, out_channels, kernel_size=3, stride=1, padding=1),#, padding_mode='circular'),
            nn.Dropout(p=1 - keep_prob))
        self.layer4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        self.layer5 = nn.Sequential(
            nn.Linear(num_states[0] * num_states[1] * out_channels, num_states[0] * num_states[1]),
            nn.ReLU()
        )
        self.layer6 = nn.Linear(num_states[0] * num_states[1], num_actions)

    def forward(self, state):
        h = state
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = torch.flatten(h,start_dim=1) # start_dim to maintain batch size
        h = self.layer5(h)
        h = self.layer6(h)
        q_values = h
        return q_values

