import torch
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
import numpy as np

# Tests if the current dqn can solve all n_examples with epsilon 0
def test_examples(n_examples, dqn, env, device, difficulty="normal", catalog=[]):
    eps = 0
    r_threshold = 0
    for i in range(n_examples):
        env.reset()
        ex = env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1]
        env.set_goal(ex)

        finish_episode = False
        while not finish_episode:
            state = env.get_state()
            state = state[None,:]

            q_o_c, curr_action = calc_q_and_take_action(dqn, state, eps, device)
            if int(curr_action) >= env.action_space.n:
                curr_action_list = catalog[int(curr_action-env.action_space.n)]
                new_state, reward, finish_episode, _ = env.step(curr_action_list) # take one step in the evironment
            else:
                new_state, reward, finish_episode, _ = env.step(curr_action)
        if not reward > r_threshold:
            print(f"Could not solve example #{i}, reward {reward} <= {r_threshold}")
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
    q_online_curr = dqn.online_model(state.to(device=device)).cpu()
    action_i = eps_greedy_policy(q_online_curr, eps) # 
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
    q_online_curr = dqn.online_model(curr_state.to(device=device)).cpu()
    with torch.no_grad():
        q_offline_next = dqn.offline_model(next_state.to(device=device)).cpu()
    
    q_target = calculate_q_targets(q_offline_next, reward, nonterminal, gamma=gamma)
    loss = dqn.calc_loss(q_online_curr, q_target, curr_action)

    return loss

def do_replay(replay_buffer, min_buffer_size, dqn, gamma, tau, batch_size, device):
    if replay_buffer.buffer_length > min_buffer_size:
        loss = sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma, device)
        dqn.optimizer.zero_grad()
        loss.backward()
        dqn.optimizer.step()

        dqn.num_online_updates += 1
        if dqn.num_online_updates % tau == 0:
            print("Update target network")
            dqn.update_target_network()

# R_avg, timed_out
def train_loop_dqn(dqn, env, replay_buffer, num_episodes, device, difficulty = "normal",
                    enable_visualization=False, batch_size=64, 
                    gamma=.94, n_examples=0, lim = 9, catalog = []):        
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    last_lim_change = 0
    init_lim = lim
    (eps, eps_end, eps_decay) = (0.99, .03, .999)
    tau = 500
    cnt_updates = 0
    R_avg = 0 # Running average of episodic rewards (total reward, disregarding discount factor)
    for i in range(num_episodes):
        if random.random() < 0.1:
            prev_eps = eps
            eps = 1.
        state = env.reset(n=n_examples, difficulty=difficulty) # Initial state
        state = state[None,:] # Add singleton dimension, to represent as batch of size 1.
        finish_episode = False # Initialize
        ep_reward = 0 # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
        steps = 0
        while not finish_episode:
            if enable_visualization:
                env.render()
            steps += 1

            # Take one step in environment. No need to compute gradients,
            # we will just store transition to replay buffer, and later sample a whole batch
            # from the replay buffer to actually take a gradient step.
            q_online_curr, curr_action = calc_q_and_take_action(dqn, state, eps, device)
            # If the action is not in the standard action space, get the action from the catalog
            if int(curr_action) >= env.action_space.n:
                curr_action_list = catalog[int(curr_action-env.action_space.n)]
                new_state, reward, finish_episode, _ = env.step(curr_action_list) # take one step in the evironment
            else:
                new_state, reward, finish_episode, _ = env.step(curr_action)
            #print(f"r:{reward},a:{curr_action}")
            new_state = new_state[None,:]
            
            nonterminal_to_buffer = not finish_episode or steps == 99
            
            # Store experienced transition to replay buffer
            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

            state = new_state
            ep_reward += reward
            # If replay buffer contains more than `batch_size` samples, perform one training step
            do_replay(replay_buffer, batch_size, dqn, gamma, tau, batch_size, device)


        # Did we do a 100% randomness episode?
        if eps == 1:
            eps = prev_eps
        else:
            eps = max(eps * eps_decay, eps_end) # decrease epsilon 
            # Current performance should not be evaluated on episodes with 100% randomness
            R_avg =  (1-gamma) * ep_reward + (gamma) * R_avg
        

        a = i
        b = ep_reward
        c = R_avg
        d = eps
        print('Episode: {:d}, Ex: {:.0f}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}'.format(a,n_examples, b,c,d))
        # If running average > 0.95 (close to 1), the task is considered solved
        if R_avg > lim/(lim+1)*0.81:
            if test_examples(n_examples, dqn, env, device, difficulty=difficulty, catalog=catalog):
                return R_avg, False
            lim += 1
            last_lim_change = i
        if i - last_lim_change > 1000:
            lim = init_lim
            last_lim_change = i
    return R_avg, True

