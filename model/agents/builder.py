from collections import namedtuple
from agents import Agent
from dqn_model import DeepQLearningModel
import torch.nn.functional as F
import torch
import numpy as np

class Builder(Agent):
    def init(self):
        # NOTE: Make the catalog actions non-symbolic
        standard_messages = [tuple(np.eye(self.max_actions(), dtype=float)[i]) for i in \
                                                            range(self.max_actions())]
        #standard_messages = [tuple(np.eye(self.num_actions + self.max_catalog_size, dtype=float)[i]) for i in range(self.num_actions)]
        for i, message in enumerate(standard_messages):
            self.symbols[message] = i
        print(self.symbols.items())

    # GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])
    def get_reward(self,message,action):
        # 1 if correct, -2 if incorrect
        return int(message == action)*3 - 2

    # transition_buffer :: [GlobalTransition]
    def append_buffer(self, transition_buffer):
        Transition = namedtuple("Transition",["s","a", "r", "next_s", "t"])
        # Skip first since we look back and skip last since it only contains next environment state
        for i in range(1,len(transition_buffer)-1):
            prev_transition = transition_buffer[i-1]
            transition      = transition_buffer[i]

            replay_transition = Transition( s       = F.one_hot(prev_transition.m,num_classes = self.num_states).float(),
                                            a       = prev_transition.a,
                                            r       = self.get_reward(prev_transition.m, prev_transition.a),
                                            next_s  = F.one_hot(transition.m,num_classes = self.num_states).float(),
                                            t       = transition.t)
            self.replay_buffer.add(replay_transition)
            
    def create_model(self, network_type):
        return DeepQLearningModel(self.device, 
                                    self.num_states, 
                                    self.max_actions(), 
                                    self.num_channels, 
                                    self.learning_rate, 
                                    network_type)
    
    def build(self, action, env):
        # Formalize action for the environment depending on if using grouped or not
        if self.grouped:
            loc = action % (self.num_actions // self.num_std_blocks)
            block = action // (self.num_actions // self.num_std_blocks)
            # num_blocks standard blocks
            if block >= self.num_std_blocks:
                block = self.catalog[block - self.num_std_blocks]
            action = (block,loc)
        else:
            if int(action) >= self.num_actions:
                action = self.catalog[int(action-self.num_actions)]
        new_state, env_reward, done, success = env.step(action)
        return (new_state, env_reward, done, success)



