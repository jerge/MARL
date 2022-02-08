import gym
#import examples as ex
import torch
import random
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay

def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        return random.randint(0,q_values.shape[1]-1)
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps):
    q_online_curr = dqn.online_model(state)
    action_i = eps_greedy_policy(q_online_curr, eps) # 

    return q_online_curr, torch.tensor(action_i)



env = gym.make('BuilderArch-v1')
env.reset()
print("------GOAL------")
env.render_state(env.goal)
print("----------------")
device = torch.device("cpu")
actions = env.action_space
num_actions = actions.n
action_list = ['Vert','Hori','Left','Right']

num_states = env.size

eps = 0

num_episodes = batch_size = gamma = learning_rate = 1 # Unnecessary variables

dqn = DeepQLearningModel(device, num_states, num_actions, learning_rate)
mod = dqn.online_model.load_state_dict(torch.load("./model1.saved"))

steps = 0
done = False
while not done:
    print("\n")
    state = env.state
    env.render()
    state = env.get_state()
    state = state[None,:]
    q_o_c, a = calc_q_and_take_action(dqn, state, eps)
    ob, r, done, _ = env.step(a)
    steps += 1
    print(list(zip(action_list,[round(x,3) for x in q_o_c.tolist()[0]])))
    print(f"Action: {action_list[a]}, Reward: {r}, New loc: {env.loc}")
print(f"\n-----RESULT----- in {steps} steps with {eps*100}% randomness")
env.render()
print("----------------")
print("------GOAL------")
env.render_state(env.goal)
print("----------------")