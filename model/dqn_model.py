import torch
from torch import nn
from collections import deque
import numpy as np


class ExperienceReplay:
    def __init__(self, device, num_states, buffer_size=1e+6):
        self._device = device
        self.__buffer = deque(maxlen=int(buffer_size))
        self._num_states = num_states

    @property
    def buffer_length(self):
        return len(self.__buffer)

    def add(self, transition):
        '''
        Adds a transition <s, a, r, s', t > to the replay buffer
        :param transition:
        :return:
        '''
        self.__buffer.append(transition)

    def sample_minibatch(self, batch_size=128):
        '''
        :param batch_size:
        :return:
        '''
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = torch.zeros([batch_size,1, self._num_states[0], self._num_states[1]],
                               dtype=torch.float)
        action_batch = torch.zeros([
            batch_size,
        ], dtype=torch.long)
        reward_batch = torch.zeros([
            batch_size,
        ], dtype=torch.float)
        nonterminal_batch = torch.zeros([
            batch_size,
        ], dtype=torch.bool)
        next_state_batch = torch.zeros([batch_size,1, self._num_states[0], self._num_states[1]],
                                    dtype=torch.float)
        for i, index in zip(range(batch_size), ids):
            state_batch[i, :] = self.__buffer[index].s
            action_batch[i] = self.__buffer[index].a
            reward_batch[i] = self.__buffer[index].r
            nonterminal_batch[i] = self.__buffer[index].t
            next_state_batch[i, :] = self.__buffer[index].next_s

        return (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            nonterminal_batch
            # torch.tensor(state_batch, dtype=torch.float, device=self._device),
            # torch.tensor(action_batch, dtype=torch.long, device=self._device),
            # torch.tensor(reward_batch, dtype=torch.float, device=self._device),
            # torch.tensor(next_state_batch,
            #              dtype=torch.float,
            #              device=self._device),
            # torch.tensor(nonterminal_batch,
            #              dtype=torch.bool,
            #              device=self._device),
        )


class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions
        keep_prob = 1
        # (Batch, Number Channels, height, width)
        out_channels = 3
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'), #num_states[0], num_states[1]
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout(p=1 - keep_prob))
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=1 - keep_prob))
        self.layer3 = nn.Linear(num_states[0] * num_states[1] * out_channels, num_states[0] * num_states[1] * out_channels)
        self.layer4 = nn.Linear(num_states[0] * num_states[1] * out_channels, num_states[0] * num_states[1])
        self.layer5 = nn.Linear(num_states[0] * num_states[1], num_actions)

        self.relu = nn.ReLU(inplace = True)
        # Initialize all bias parameters to 0, according to old Keras implementation
        #nn.init.zeros_(self._fc1.bias)
        #nn.init.zeros_(self._fc2.bias)
        #nn.init.zeros_(self._fc_final.bias)
        # Initialize final layer uniformly in [-1e-6, 1e-6] range, according to old Keras implementation
        #nn.init.uniform_(self._fc_final.weight, a=-1e-6, b=1e-6)

    def forward(self, state):
        #print(state.shape)
        h = self.layer1(state)
        #print(h.shape)
        h = self.layer2(h)
        #print(h.shape)
        h = torch.flatten(h,start_dim=1) # start_dim to maintain batch size
        #print(h.shape)
        h = self.layer3(h)
        #print(h.shape)
        h = self.relu(h)
        h = self.layer4(h)
        #print(h.shape)
        h = self.relu(h)
        h = self.layer5(h)
        #print(h.shape)
        q_values = h
        return q_values


class DeepQLearningModel(object):
    def __init__(self, device, num_states, num_actions, learning_rate):
        self._device = device
        self._num_states = num_states
        self._num_actions = num_actions
        self._lr = learning_rate

        # Define the two Q-networks
        self.online_model = QNetwork(self._num_states,
                                     self._num_actions).to(device=self._device)
        self.offline_model = QNetwork(
            self._num_states, self._num_actions).to(device=self._device)

        # Define optimizer. Should update online network parameters only.
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(),
                                             lr=self._lr)

        # Define loss function
        self._mse = nn.MSELoss(reduction='mean').to(device=self._device)

    def calc_loss(self, q_online_curr, q_target, a):
        '''
        Calculate loss for given batch
        :param q_online_curr: batch of q values at current state. Shape (N, num actions)
        :param q_target: batch of temporal difference targets. Shape (N,)
        :param a: batch of actions taken at current state. Shape (N,)
        :return:
        '''
        batch_size = q_online_curr.shape[0]
        assert q_online_curr.shape == (batch_size, self._num_actions)
        assert q_target.shape == (batch_size, )
        assert a.shape == (batch_size, )

        # Select only the Q-values corresponding to the actions taken (loss should only be applied for these)
        q_online_curr_allactions = q_online_curr
        q_online_curr = q_online_curr[torch.arange(batch_size),
                                      a]  # New shape: (batch_size,)
        assert q_online_curr.shape == (batch_size, )
        for j in [0, 3, 4]:
            assert q_online_curr_allactions[j, a[j]] == q_online_curr[j]

        # Make sure that gradient is not back-propagated through Q target
        assert not q_target.requires_grad

        loss = self._mse(q_online_curr, q_target)
        assert loss.shape == ()

        return loss

    def update_target_network(self):
        '''
        Update target network parameters, by copying from online network.
        '''
        online_params = self.online_model.state_dict()
        self.offline_model.load_state_dict(online_params)
