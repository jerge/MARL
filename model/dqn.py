import torch
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
#import examples as ex
import numpy as np

# Tests if the current dqn can solve all n_examples with epsilon 0
def test_examples(n_examples, dqn, env, device, difficulty="normal"):
    eps = 0
    for i in range(n_examples):
        env.reset()
        ex = env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1]
        env.set_goal(ex)

        done = False
        while not done:
            state = env.get_state()
            state = state[None,:]

            q_o_c, a = calc_q_and_take_action(dqn, state, eps, device)
            ob, r, done, _ = env.step(a)
        if not r >= 0.9:
            print(f"Could not solve examples {i}")
            return False
    print(f"Solved all {n_examples} examples")
    return True


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

def calc_q_and_take_action(dqn, state, eps, device):
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
    q_online_curr = dqn.online_model(state.to(device=device)).cpu()
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
def sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma, device):
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
    q_online_curr = dqn.online_model(curr_state.to(device=device)).cpu()
    with torch.no_grad():
        q_offline_next = dqn.offline_model(next_state.to(device=device)).cpu()
    
    q_target = calculate_q_targets(q_offline_next, reward, nonterminal, gamma=gamma)
    loss = dqn.calc_loss(q_online_curr, q_target, curr_action)

    return loss

# R_buf, R_avg, timedout?
def train_loop_dqn(dqn, env, replay_buffer, num_episodes, device, difficulty = "normal",
                    enable_visualization=False, batch_size=64, 
                    gamma=.94, n_examples=0, lim = 9):        
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    eps = 0.99
    eps_end = .03
    eps_decay = .999
    tau = 500
    cnt_updates = 0
    R_buffer = []
    R_avg = []
    for i in range(num_episodes):
        if random.random() < 0.1:
            prev_eps = eps
            eps = 1.
        state = env.reset(n=n_examples, difficulty=difficulty) # Initial state
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
            q_online_curr, curr_action = calc_q_and_take_action(dqn, state, eps, device)
            q_buffer.append(q_online_curr)
            new_state, reward, finish_episode, _ = env.step(curr_action) # take one step in the evironment
            #print(f"r:{reward},a:{curr_action}")
            new_state = new_state[None,:]
            
            nonterminal_to_buffer = not finish_episode or steps == 99
            
            # Store experienced transition to replay buffer
            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

            state = new_state
            ep_reward += reward
            # If replay buffer contains more than ? samples, perform one training step
            if replay_buffer.buffer_length > batch_size:
                loss = sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma, device)
                dqn.optimizer.zero_grad()
                loss.backward()
                dqn.optimizer.step()

                cnt_updates += 1
                if cnt_updates % tau == 0:
                    print(f"Update target network, using {n_examples} examples")
                    dqn.update_target_network()
                
        if eps == 1:
            eps = prev_eps
        eps = max(eps * eps_decay, eps_end) # decrease epsilon
        # if eps < 0.20:
        #     eps_decay = 0.9999      
        R_buffer.append(ep_reward)
        
        # Running average of episodic rewards (total reward, disregarding discount factor)
        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i-1]) if i > 0 else R_avg.append(R_buffer[i])
        a = i
        b = ep_reward
        c = R_avg[-1]
        d = eps
        e = np.mean([torch.mean(q).cpu().detach().numpy() for q in q_buffer])
        print('Episode: {:d}, Ex: {:.0f}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Avg Q: {:.4g}'.format(a,n_examples, b,c,d,e))
        # If running average > 0.95 (close to 1), the task is considered solved
        if R_avg[-1] > lim/(lim+1):
            if test_examples(n_examples, dqn, env, device, difficulty=difficulty):
                return R_buffer, R_avg, False
            lim +=1
            #return R_buffer, R_avg
    return R_buffer, R_avg, True

