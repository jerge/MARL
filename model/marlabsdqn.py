import torch
import torch.nn.functional as F
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple, deque
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
import numpy as np

# Tests if the current dqn can solve all n_examples with epsilon 0
def test_examples(n_examples, architect, builder, env, device, difficulty="normal"):
    eps = 0
    r_threshold = 0
    for i in range(n_examples):
        env.reset()
        # Get the i:th example and set it as goal
        ex = env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1]
        env.set_goal(ex)

        done = False
        while not done:
            state = env.get_state()[None,:]
            # A: Generate message from state
            q_a, message_one_hot = calc_q_and_take_action(architect.dqn, state, eps, device)
            message = torch.argmax(message_one_hot)
            message_one_hot = message_one_hot[None,:]
            
            # B: Generate action from message
            q_b, action_one_hot = calc_q_and_take_action(builder.dqn, message_one_hot, eps, device, debug=False) # eps
            action = torch.argmax(action_one_hot)
            action_one_hot = action_one_hot[None,:]

            # Env: Take action
            (new_state, reward, done) = builder.build(action, env)
            new_state = new_state[None,:]
        if not reward > r_threshold:
            print(f"Could not solve example #{i}")
            return False
    print(f"Solved all {n_examples} examples. Reward threshold >{r_threshold}")
    return True

def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        r = random.randint(0,q_values.shape[1]-1)
        #print(r)
        return r
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps, device, debug = False):
    q_online_curr = dqn.online_model(state.to(device=device)).cpu()
    action_i = torch.tensor(eps_greedy_policy(q_online_curr, eps))
    if debug:
        print(q_online_curr)
    return q_online_curr, F.one_hot(action_i,num_classes = q_online_curr.shape[1]).float()

def calculate_q_targets(q1_batch, r_batch, nonterminal_batch, gamma=.99):
    Y = r_batch + nonterminal_batch.long() * gamma * (torch.max(q1_batch,dim=1)[0])
    return Y

def sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma, device):
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

def train_loop(env, architect, builder, n_episodes, 
                device, n_examples, difficulty,
                batch_size = 128, lim = 9):
    GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])
    min_buffer_size = 100
    (eps, eps_decay, eps_end) = (0.99, 0.999, 0.03)
    (last_lim_change, init_lim) = (0, lim)
    tau = 500 # Frequency of target network updates
    R_avg = 0 # Running average of episodic rewards (total reward, disregarding discount factor)
    tot_steps = 0
    trial = False

    episode_buffer = deque(maxlen=200) # queue of entire episodes
    for i in range(n_episodes):
        state = env.reset(n=n_examples, difficulty=difficulty)[None,:] # Initial state with added batch size

        (ep_reward, steps, done) = (0, 0, False)
        episode_history = [] # list of all 'GlobalTransition's in the current episode
        while not done:
            steps += 1
            tot_steps += 1
            if tot_steps % 1000 == 0 and tot_steps != 0:# and i != 0:
                trial = True
                architect.training = architect.training != True
                builder.training   = builder.training   != True
            
            state = env.get_state()[None,:]
            # A: Generate message from state
            if architect.training:
                _, message_one_hot = calc_q_and_take_action(architect.dqn, state, eps, device)
            else:
                with torch.no_grad():
                    _, message_one_hot = calc_q_and_take_action(architect.dqn, state, eps_end, device)
            message = torch.argmax(message_one_hot)
            message_one_hot = message_one_hot[None,:]
            
            # B: Generate action from message
            if builder.training:
                q, action_one_hot = calc_q_and_take_action(builder.dqn, message_one_hot, eps, device, debug=False) # eps
            else:
                with torch.no_grad():
                    _, action_one_hot = calc_q_and_take_action(builder.dqn, message_one_hot, 0.03, device)
            action = torch.argmax(action_one_hot)
            action_one_hot = action_one_hot[None,:]

            # Env: Take action
            (new_state, reward, done) = builder.build(action, env)
            new_state = new_state[None,:]

            # Save transition
            nonterminal_to_buffer = not done or steps == 99
            episode_history.append(GlobalTransition(s=state, m = message, a = action, r = reward, t = nonterminal_to_buffer))

            ep_reward += reward
        # Extra transition for architect
        episode_history.append(GlobalTransition(s=new_state, m = None, a = None, r = None, t = None))
        
        # Add the episode to the replay buffers
        architect.append_buffer(episode_history)
        builder.append_buffer(episode_history)
        # Add the episode, if successful, to the episode buffer for later abstraction
        if done and reward > 0:
            episode_buffer.extend(episode_history)
        
        do_replay(architect.replay_buffer, min_buffer_size, architect.dqn, architect.gamma, tau, batch_size, device)
        do_replay(builder.replay_buffer,   min_buffer_size, builder.dqn,   builder.gamma,   tau, batch_size, device)

        eps = max(eps * eps_decay, eps_end)

        R_avg =  (1-architect.gamma) * ep_reward + (architect.gamma) * R_avg
        t = "a,b" if architect.training and builder.training else "a" if architect.training else "b" if builder.training else "none"
        print('Episode: {:d}, Ex: {:.0f}, Steps:{: 4d}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Trainee: {}'.format(
                                                                                    i, n_examples, steps, ep_reward, R_avg, eps, t))

        #if R_avg > lim/(lim+1):
        if trial:
            print("---------------------------------")
            if test_examples(n_examples, architect, builder, env, device, difficulty=difficulty):
                return R_avg, False
            print("---------------------------------")
            lim += 1
            trial = False
            last_lim_change = i
        if i - last_lim_change > 1000:
            lim = init_lim
            last_lim_change = i
    return R_avg, True