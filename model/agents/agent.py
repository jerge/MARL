from collections import namedtuple
from dqn_model import ExperienceReplay
from abc import ABC, abstractmethod

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
        


    def increase_catalog(self, item):
        assert len(catalog) < max_catalog_size, f"The catalog {catalog} is too big and cannot be increased, max_size = {max_catalog_size}"
        catalog.append(item)


# batch_size = 128
# gamma = .95
# learning_rate = 1e-4

    @abstractmethod
    def create_model(self, network_type):
        pass