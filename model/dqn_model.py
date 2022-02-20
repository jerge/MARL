import torch
from torch import nn
from collections import deque
import numpy as np
import networks

class ExperienceReplay:
    def __init__(self, device, num_states, buffer_size=1e+6, input_channels=1):
        self._device = device
        self.__buffer = deque(maxlen=int(buffer_size))
        self._num_states = num_states
        self._input_channels = input_channels

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
        if type(self._num_states) == int:
            state_batch = torch.zeros([batch_size,self._input_channels, self._num_states],
                               dtype=torch.float)
        else:
            state_batch = torch.zeros([batch_size,self._input_channels, self._num_states[0], self._num_states[1]],
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

        if type(self._num_states) == int:
            next_state_batch = torch.zeros([batch_size,self._input_channels, self._num_states],
                               dtype=torch.float)
        else:
            next_state_batch = torch.zeros([batch_size,self._input_channels, self._num_states[0], self._num_states[1]],
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
        )


class DeepQLearningModel(object):
    def __init__(self, device, num_states, num_actions, num_channels, learning_rate, network_type):
        self._device = device
        self._num_states = num_states
        self._num_actions = num_actions
        self._num_channels = num_channels
        self._lr = learning_rate
        self.num_online_updates = 0

        if network_type.lower() == "dense":
            self.online_model = networks.DenseNetwork(self._num_states,
                                     self._num_actions,
                                     self._num_channels).to(device=self._device)
            self.offline_model = networks.DenseNetwork(self._num_states, 
                                      self._num_actions,
                                      self._num_channels).to(device=self._device)
        elif network_type.lower() == "conv" or network_type.lower() == "convolutional":
            self.online_model = networks.ConvNetwork(self._num_states,
                                     self._num_actions,
                                     self._num_channels).to(device=self._device)
            self.offline_model = networks.ConvNetwork(self._num_states, 
                                      self._num_actions,
                                      self._num_channels).to(device=self._device)
        else:
            assert False, f"Invalid network type {network_type}"


        # Define the two Q-networks
        
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
