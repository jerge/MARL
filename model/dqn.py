import torch
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
#import examples as ex
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def eps_greedy_policy(q_values, eps):
    '''
    Creates an epsilon-greedy policy
    :param q_values: set of q-values of shape (num actions,)
    :param eps: probability of taking a uniform random action 
    :return: action_index
    '''
    if random.random() < eps:
        return random.randint(0,q_values.shape[1]-1)
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps):
    '''
    Calculate Q-values for current state, and take an action according to an epsilon-greedy policy.
    Inputs:
        dqn   - DQN model. An object holding the online / offline Q-networks, and some related methods.
        state  - Current state. Numpy array, shape (1, num_states).
        eps    - Exploration parameter.
    Returns:
        q_online_curr   - Q(s,a) for current state s. Numpy array, shape (1, num_actions) or  (num_actions,).
        curr_action     - Selected action (0 or 1, i.e., left or right), sampled from epsilon-greedy policy. Integer.
    '''
    # dqn.online_model & dqn.offline_model are Pytorch modules for online / offline Q-networks, 
    #   which take the state as input, and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).
    #state = torch.from_numpy(state)
    q_online_curr = dqn.online_model(state)
    #q_o_c = q_online_curr.cpu().detach().numpy().reshape((-1,))
    action_i = eps_greedy_policy(q_online_curr, eps) # 
    #curr_action = np.random.choice([0,1],size=1,p=actions)[0]
    return q_online_curr, torch.tensor(action_i)

def calculate_q_targets(q1_batch, r_batch, nonterminal_batch, gamma=.99):
    '''
    Calculates the Q target used for the loss
    : param q1_batch: Batch of q_hat(s', a) from target network. FloatTensor, shape (N, num actions)
    : param r_batch: Batch of rewards. FloatTensor, shape (N,)
    : param nonterminal_batch: Batch of booleans, with False elements if state s' is terminal and True otherwise. BoolTensor, shape (N,)
    : param gamma: Discount factor, float.
    : return: Q target. FloatTensor, shape (N,)
    '''
    # target = y_i = r_i +  gamma_a{max}^q(siprime, a; offline_param), 

    Y = r_batch + nonterminal_batch.long() * gamma * (torch.max(q1_batch,dim=1)[0])
    return Y

# Answer:
def sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma):
    '''
    Sample mini-batch from replay buffer, and compute the mini-batch loss
    Inputs:
        dqn          - DQN model. An object holding the online / offline Q-networks, and some related methods.
        replay_buffer - Replay buffer object (from which samples will be drawn)
        batch_size    - Batch size
        gamma         - Discount factor
    Returns:
        Mini-batch loss, on which .backward() will be called to compute gradient.
    '''
    # Sample a minibatch of transitions from replay buffer
    curr_state, curr_action, reward, next_state, nonterminal = replay_buffer.sample_minibatch(batch_size)

    # FYI:
    # dqn.online_model & dqn.offline_model are Pytorch modules for online / offline Q-networks, which take the state as input, and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).

    # YOUR CODE HERE
    q_online_curr = dqn.online_model(curr_state)
    with torch.no_grad():
        q_offline_next = dqn.offline_model(next_state)
    
    q_target = calculate_q_targets(q_offline_next, reward, nonterminal, gamma=gamma)
    loss = dqn.calc_loss(q_online_curr, q_target, curr_action)

    return loss

def train_loop_dqn(dqn, env, replay_buffer, num_episodes, enable_visualization=False, batch_size=64, gamma=.94):        
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    eps = 1.
    eps_end = .03
    eps_decay = .999
    tau = 1000
    cnt_updates = 0
    R_buffer = []
    R_avg = []
    for i in range(num_episodes):
        state = env.reset() # Initial state
        state = state[None,:] # Add singleton dimension, to represent as batch of size 1.
        finish_episode = False # Initialize
        ep_reward = 0 # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
        q_buffer = []
        steps = 0
        while not finish_episode:
            if enable_visualization:
                env.render()
            steps += 1

            # Take one step in environment. No need to compute gradients,
            # we will just store transition to replay buffer, and later sample a whole batch
            # from the replay buffer to actually take a gradient step.
            q_online_curr, curr_action = calc_q_and_take_action(dqn, state, eps)
            q_buffer.append(q_online_curr)
            new_state, reward, finish_episode, _ = env.step(curr_action) # take one step in the evironment
            #print(f"r:{reward},a:{curr_action}")
            new_state = new_state[None,:]
            
            # Assess whether terminal state was reached.
            # The episode may end due to having reached 50 steps, 
            # but we should not regard this as reaching the terminal state, 
            # and hence not disregard Q(s',a) from the Q target.
            nonterminal_to_buffer = not finish_episode or steps == 99
            
            # Store experienced transition to replay buffer
            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

            state = new_state
            ep_reward += reward
            # If replay buffer contains more than ? samples, perform one training step
            if replay_buffer.buffer_length > batch_size:
                loss = sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma)
                dqn.optimizer.zero_grad()
                loss.backward()
                dqn.optimizer.step()

                cnt_updates += 1
                if cnt_updates % tau == 0:
                    dqn.update_target_network()
                
        eps = max(eps * eps_decay, eps_end) # decrease epsilon
        if eps < 0.25:
            eps_decay = 0.9999      
        R_buffer.append(ep_reward)
        
        # Running average of episodic rewards (total reward, disregarding discount factor)
        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i-1]) if i > 0 else R_avg.append(R_buffer[i])
        a = i
        b = ep_reward
        c = R_avg[-1]
        d = eps
        #print(type(np.array(q_buffer)))
        e = np.mean([torch.mean(q).detach().numpy() for q in q_buffer])
        print('Episode: {:d}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Avg Q: {:.4g}'.format(a,b,c,d,e))        
        # If running average > 195 (close to 200), the task is considered solved
        if R_avg[-1] > 195:
            return R_buffer, R_avg
    return R_buffer, R_avg

# Create the environment
env = gym.make('BuilderArch-v1')
#ex1 = ex.get_examples7()[1]
#print(ex1)
env.reset()
#env.set_goal(ex1)
# Enable visualization? Does not work in all environments.
enable_visualization = False

# Initializations
actions = env.action_space
num_actions = actions.n
num_states = env.size
input_channels = 2

num_episodes = 100
batch_size = 128
gamma = .90
learning_rate = 1e-4

# Object holding our online / offline Q-Networks
dqn = DeepQLearningModel(device, num_states, num_actions, learning_rate)
#dqn.online_model.load_state_dict(torch.load("./model1.saved"))
#dqn.offline_model.load_state_dict(torch.load("./model1.saved"))
# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored 
# for training
replay_buffer = ExperienceReplay(device, num_states, input_channels=input_channels)

# Train
R, R_avg = train_loop_dqn(dqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma)

torch.save(dqn.online_model.state_dict(), "./model3.saved")
