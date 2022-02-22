import torch
import torch.nn.functional as F
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
#import examples as ex
import numpy as np

def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        return random.randint(0,q_values.shape[1]-1)
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps, device):
    q_online_curr = dqn.online_model(state.to(device=device)).cpu()
    action_i = torch.tensor(eps_greedy_policy(q_online_curr, eps))
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
        #print(dqn.num_online_updates)
        dqn.num_online_updates += 1
        if dqn.num_online_updates % tau == 0:
            print("Update target network")
            dqn.update_target_network()

def train_loop(env, architect, builder, n_episodes, a_replay_buffer, b_replay_buffer, device, n_examples, difficulty, 
                batch_size = 128, training_architect = True, training_builder = True):
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    min_buffer_size = 100
    gamma = 0.95
    (eps, eps_decay, eps_end) = (0.99, 0.9999, 0.03)

    tau = 100 # Frequency of architect target network updates
    R_avg = 0 # Running average of episodic rewards (total reward, disregarding discount factor)
    for i in range(n_episodes):
        if i % 50 == 0:
            training_architect = training_architect != True
            training_builder   = training_builder   != True
        state_with_goal = env.reset(n=n_examples, difficulty=difficulty) # Initial state
        state_with_goal = state_with_goal[None,:]
        
        ep_reward = 0
        message = -1 # -1 is the token corresponding to "first action"
        done = False
        while not done:
            old_message = message

            if training_architect:
                _, message_one_hot = calc_q_and_take_action(architect, state_with_goal, eps, device)
            else:
                with torch.no_grad():
                    _, message_one_hot = calc_q_and_take_action(architect, state_with_goal, 0.03, device)
            message = torch.argmax(message_one_hot)
            message_one_hot = message_one_hot[None,:]
            
            # Should the builder get the state?
            if training_builder:
                _, action_one_hot = calc_q_and_take_action(builder, message_one_hot, eps, device)
            else: 
                with torch.no_grad():
                    _, action_one_hot = calc_q_and_take_action(builder, message_one_hot, 0.03, device)
            action = torch.argmax(action_one_hot)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state[None,:]
            
            if training_architect:
                a_replay = Transition(s=state_with_goal, a=message, r=reward, next_s=new_state, t=done)
                a_replay_buffer.add(a_replay)
                do_replay(a_replay_buffer, min_buffer_size, architect, gamma, tau, batch_size, device)

            if training_builder:
                # TODO: What reward should the builder get
                b_reward = int(action == message)
                # TODO: Does the builder need one extra iteration?
                if not old_message == -1:
                    b_replay = Transition(s=old_message, a=action,  r=b_reward, next_s=message, t=done)
                    b_replay_buffer.add(b_replay)
                do_replay(b_replay_buffer, min_buffer_size, builder,   gamma, tau, batch_size, device)

            ep_reward += reward
            state_with_goal = new_state
        eps = max(eps * eps_decay, eps_end)

        R_avg =  (1-gamma) * ep_reward + (gamma) * R_avg
        t = "a" if training_architect else "b"
        print('Episode: {:d}, Ex: {:.0f}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Trainee: {}'.format(
                                                                                    i, n_examples, ep_reward, R_avg, eps, t))