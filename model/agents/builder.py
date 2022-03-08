from collections import namedtuple, Counter
from agents import Agent
from dqn_model import DeepQLearningModel
class Builder(Agent):
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


            replay_transition = Transition( s       = prev_transition.m,
                                            a       = prev_transition.a,
                                            r       = self.get_reward(prev_transition.m, prev_transition.a),
                                            next_s  = transition.m,
                                            t       = transition.t)
            self.replay_buffer.add(replay_transition)
            
    def create_model(self, network_type):
        return DeepQLearningModel(self.device, 
                                    self.num_states, 
                                    self.num_actions + self.max_catalog_size, 
                                    self.num_channels, 
                                    self.learning_rate, 
                                    network_type)

    # Checks through the entire replay_buffer to see if a symbol has been properly learnt
    def learn_symbol(self):
        threshold = 0.95
        # state -> Counter() :: action -> int
        state_dict = dict()
        # IS TRANSIOTON.S a proper key?
        for transition in self.replay_buffer.sample_minibatch(batch_size=10000):
            # Skip if already learnt
            s = tuple(transition.s.tolist())
            a = tuple(transition.a.tolist())
            if s in self.symbols.keys():
                continue
            if state_dict[s] == None:
                state_dict[s] = Counter()
            state_dict[s][a] += 1
        for s, counter in state_dict.items():
            total = sum(counter.values())
            for a, amount in counter.items():
                if amount / total > threshold:
                    self.symbols[s] = a 
                    print(f"Added {a} to symbol_list for state {s}")


    
    def build(self, action, env):
        if int(action) >= env.action_space.n:
            action = self.catalog[int(action-env.action_space.n)]
        new_state, env_reward, done, success = env.step(action)
        return (new_state, env_reward, done, success)



