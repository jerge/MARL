from collections import namedtuple
from agents import Agent
from dqn_model import DeepQLearningModel
class Architect(Agent):
    # GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])

    # transition_buffer :: [GlobalTransition]
    def append_buffer(self, transition_buffer):
        Transition = namedtuple("Transition",["s", "a", "r", "next_s", "t"])
        for i in range(0,len(transition_buffer)-1):
            transition = transition_buffer[i]
            next_transition = transition_buffer[i+1]

            replay_transition = Transition( s       = transition.s,
                                            a       = transition.m,
                                            r       = transition.r,
                                            next_s  = next_transition.s,
                                            t       = transition.t)
            self.replay_buffer.add(replay_transition)
    
    def create_model(self, network_type):
        return DeepQLearningModel(self.device, 
                                    self.num_states, 
                                    self.num_actions + self.max_catalog_size, 
                                    self.num_channels, 
                                    self.learning_rate, 
                                    network_type)

            
    def init(self):
        pass






