import gym
#import examples as ex
import torch
import torch.nn.functional as F
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

def calc_q_and_take_action(dqn, state, eps, device):
    q_online_curr = dqn.online_model(state.to(device=device)).cpu()
    action_i = torch.tensor(eps_greedy_policy(q_online_curr, eps))
    return q_online_curr, F.one_hot(action_i,num_classes = q_online_curr.shape[1]).float()

def test_examples(n_examples, dqn, env, difficulty="normal"):
    eps = 0
    r_threshold = 0
    for i in range(n_examples):
        env.reset()
        env.set_goal(env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1])

        done = False
        while not done:
            state = env.get_state()
            state = state[None,:]

            q_o_c, a = calc_q_and_take_action(dqn, state, eps)
            ob, r, done, _ = env.step(a)
        if not r > r_threshold:
            print(f"Could not solve examples {i}, reward {r} <= {r_threshold}")
            return False
    print(f"Solved all {n_examples} examples")
    return True

device = torch.device("cpu")
env = gym.make('BuilderArch-v1')
n = 16
env.reset(n=n)

print("------GOAL------")
env.render_state(env.goal)
print("----------------")
device = torch.device("cpu")

actions = env.action_space
num_actions = actions.n
catalog = [torch.tensor([3,3]),torch.tensor([2,2]),torch.tensor([3,0]),torch.tensor([3,1]),torch.tensor([1,1])]
action_list = ['Vert','Hori','Left','Right']
catalog_names = [",".join([action_list[item][0] for item in itemlist]) for itemlist in [items.tolist() for items in catalog]]
action_list = action_list + catalog_names
num_messages = num_actions + len(catalog)

num_states = env.size

eps = 0

num_episodes = batch_size = gamma = learning_rate = 1 # Unnecessary variables

#TODO: Temp
network_type = "dense"

a_dqn = DeepQLearningModel(device, num_states[0] * num_states[1], num_messages, 2, learning_rate, network_type)
b_dqn = DeepQLearningModel(device, num_messages, num_actions + len(catalog), 1, learning_rate, "small" + network_type)

interrupted = "_interrupted" # set to empty string empty to not use interrupted
a_dqn.online_model.load_state_dict(torch.load(f"./model_checkpoints/{env.size[0]}marlcnormala{n-1}{interrupted}.saved"))
b_dqn.online_model.load_state_dict(torch.load(f"./model_checkpoints/{env.size[0]}marlcnormalb{n-1}{interrupted}.saved"))

steps = 0
done = False
while not done:
    print("\n")
    env.render()
    state = env.get_state()
    state = state[None,:]
    #print(state)
    a_q_values, message_one_hot = calc_q_and_take_action(a_dqn, state, eps, device)
    message = torch.argmax(message_one_hot)
    message_one_hot = message_one_hot[None,:]
    b_q_values, action_one_hot =  calc_q_and_take_action(b_dqn, message_one_hot, eps, device)
    action = torch.argmax(action_one_hot)
    if int(action) >= env.action_space.n:
        action_tensor = catalog[int(action-env.action_space.n)]
        new_state, reward, done, _ = env.step(action_tensor) # take one step in the evironment
    else:
        new_state, reward, done, _ = env.step(action)
    steps += 1
    print(list(zip(action_list,[round(x,3) for x in a_q_values.tolist()[0]])))
    print(list(zip(action_list,[round(x,3) for x in b_q_values.tolist()[0]])))
    print(f"Message: {action_list[message]}, Action: {action_list[action]}, Reward: {reward}, New loc: {env.loc}")
print(f"\n-----RESULT----- in {steps} steps with {eps*100}% randomness")
env.render()
print("----------------")
print("------GOAL------")
env.render_state(env.goal)
print("----------------")

#TODO: test_examples(n,dqn,env)