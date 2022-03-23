from collections import namedtuple, Counter
from dqn_model import ExperienceReplay
from abc import ABC, abstractmethod
import torch

class Agent(ABC):
    GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])

    def __init__(self, num_states, num_actions, num_channels, device, network_type, 
                    catalog = [], max_catalog_size = 0, gamma = 0.95, learning_rate = 1e-4, training = False, grouped = False):
        self.num_states = num_states if type(num_states) == int or "conv" in network_type else num_states[0] * num_states[1]
        self.grouped = grouped
        self.num_actions = num_actions
        self.num_std_blocks = 2
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
        
        self.init()
    
    @abstractmethod
    def init(self):
        pass

    # Returns the number of currently available acitons
    def available_actions(self):
        if self.grouped:
            return (self.num_std_blocks+len(self.catalog)) * (self.num_actions // self.num_std_blocks)
        else:
            return self.num_actions + len(self.catalog)

    # Returns the number of possibly available actions
    def max_actions(self):
        if self.grouped:
            return (self.num_std_blocks+self.max_catalog_size) * (self.num_actions // self.num_std_blocks)
        else:
            return self.num_actions + self.max_catalog_size


    # Returns either the learnt symbolic actions for the input or
    # False if the symbol is not learnt
    def use_symbol(self, inp):
        x = self.symbols.get(tuple(inp[0].tolist()), None)
        return x


    # Checks through the entire replay_buffer to see if a symbol has been properly learnt
    def learn_symbol(self):
        threshold = 0.95
        # state -> Counter() :: action -> int
        state_dict = dict()
        
        batch_size = 10000
        if self.replay_buffer.buffer_length < batch_size:
            return
        rb = self.replay_buffer.sample_minibatch(batch_size=batch_size)
        for i in range(batch_size):
            s = rb[0][i][0]
            a = rb[1][i]
            # Skip if already learnt
            s = tuple(s.tolist())
            a = int(a)#a.tolist())
            if s in self.symbols.keys():
                continue
            elif state_dict.get(s,None) == None:
                state_dict[s] = Counter()
            state_dict[s][a] += 1
        for s, counter in state_dict.items():
            total = sum(counter.values())
            for a, amount in counter.items():
                if amount / total > threshold:
                    self.symbols[s] = a 
                    print(f"Added {a} to symbol_list for state {s}")

    # TODO: fixe grouped
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

    def catalog_names(self, grouped = False):
        assert not grouped, "grouped not implemented"
        action_list = ['Vert','Hori','Left','Right']
        catalog_names = action_list + [",".join([action_list[item][0] for item in itemlist]) for itemlist in [items.tolist() for items in self.catalog]]
    
    @abstractmethod
    def create_model(self, network_type):
        pass