from collections import namedtuple
from dqn_model import ExperienceReplay
from abc import ABC, abstractmethod
import torch

class Agent(ABC):
    GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])

    def __init__(self, num_states, num_actions, num_channels, device, network_type, 
                    catalog = [], max_catalog_size = 0, gamma = 0.95, learning_rate = 1e-4, training = False):
        self.num_states = num_states if type(num_states) == int else num_states[0] * num_states[1]
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.device = device
        self.training = training
    
        self.replay_buffer = ExperienceReplay(device, num_states, input_channels=num_channels)
        assert len(catalog) <= max_catalog_size, f"The initial catalog {catalog} is bigger than max_size {max_catalog_size}"
        self.catalog = catalog
        self.max_catalog_size = max_catalog_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.dqn = self.create_model(network_type)

        self.symbols = dict()
    
    # Returns either the learnt symbolic actions for the input or
    # false if the symbol is not learnt
    def use_symbol(self, inp):
        return self.symbols.get(tuple(inp.tolist()), False)

    def to_primitives(self, sequence):
        # sequence :: String
        primitive_sequence = []
        for a in [int(a) for a in sequence]:
            primitive_sequence.extend(self.catalog[a-self.num_actions]) if a >= self.num_actions \
                else primitive_sequence.append(a)
        return primitive_sequence

    def increase_catalog(self, sequence):
        # sequence :: String
        if not len(self.catalog) < self.max_catalog_size:
            print(f"The catalog {self.catalog} is too big and cannot be increased, max_size = {self.max_catalog_size}")
            return
        
        primitive_sequence = self.to_primitives(sequence)
        if not primitive_sequence in [c.tolist() for c in self.catalog]:
            self.catalog.append(torch.tensor(primitive_sequence))

    @abstractmethod
    def create_model(self, network_type):
        pass