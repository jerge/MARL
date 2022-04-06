import torch
from torch import nn
from torchvision import transforms

# [Goal,State] -> num_actions
class DenseNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_channels):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        # (Batch, Number Channels, height, width)
        self.layer1 = nn.Sequential(
            nn.Linear(num_states * num_channels, num_states * 16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 16),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 16),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 16),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(num_states * 16, num_states),
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0)#.to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 

class DenseNoiseNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_channels):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        # (Batch, Number Channels, height, width)
        self.layer1 = nn.Sequential(
            nn.Linear(num_states * num_channels, num_states * 16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 16),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 16),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 16),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(num_states * 16, num_states),
            nn.ReLU())
        self.layer6 = nn.Linear(num_states, num_actions)

        self.blur = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2))

    def forward(self, state):
        h = self.blur(state)
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
class SmallDenseNoiseNetwork(nn.Module):
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
            nn.ReLU(),
            #nn.BatchNorm1d(num_states * 4),
            GaussianNoise(0.1))
        self.layer3 = nn.Linear(num_states*4, num_actions)

    def forward(self, state):
        h = state
        h = torch.flatten(h,start_dim=1) # start_dim to maintain batch size
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        q_values = h
        return q_values

# [Goal,State] -> num_actions
class BigDenseNetwork(nn.Module):
    def __init__(self, num_states, num_actions, num_channels):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        # (Batch, Number Channels, height, width)
        self.layer1 = nn.Sequential(
            nn.Linear(num_states * num_channels, num_states * 16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(num_states * 16, num_states * 256),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Linear(num_states * 256, num_states * 512),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(num_states * 512, num_states * 64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Linear(num_states * 64, num_states * 8),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Linear(num_states * 8, num_states * 1),
            nn.ReLU())
        self.layer7 = nn.Linear(num_states, num_actions)

    def forward(self, state):
        h = state
        h = torch.flatten(h,start_dim=1) # start_dim to maintain batch size
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.layer6(h)
        h = self.layer7(h)
        q_values = h
        return q_values
