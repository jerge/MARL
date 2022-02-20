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

        dqn.num_online_updates += 1
        if dqn.num_online_updates % tau == 0:
            print("Update target network")
            dqn.update_target_network()

def train_loop(env, architect, builder, n_episodes, a_replay_buffer, b_replay_buffer, device, n_examples, difficulty, batch_size = 128):
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    min_buffer_size = 100
    gamma = 0.95
    eps = 0.99
    eps_decay = 0.999
    eps_end = 0.03

    tau = 100 # Frequency of architect target network updates
    for i in range(n_episodes):
        state_with_goal = env.reset(n=n_examples, difficulty=difficulty) # Initial state
        state_with_goal = state_with_goal[None,:]

        message = -1 # -1 is the token corresponding to "first action"
        done = False
        while not done:
            old_message = message

            _, message_one_hot = calc_q_and_take_action(architect, state_with_goal, eps, device)
            message = torch.argmax(message_one_hot)
            message_one_hot = message_one_hot[None,:]
            
            # Should the builder get the state?
            _, action_one_hot = calc_q_and_take_action(builder, message_one_hot, eps, device)
            action = torch.argmax(action_one_hot)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state[None,:]
            
            
            a_replay = Transition(s=state_with_goal, a=message, r=reward, next_s=new_state, t=done)
            a_replay_buffer.add(a_replay)

            # TODO: What reward should the builder get
            # TODO: Does the builder need one extra iteration?
            if not old_message == None:
                b_replay = Transition(s=old_message, a=action,  r=reward, next_s=message, t=done)
                b_replay_buffer.add(b_replay)
            

            do_replay(a_replay_buffer, min_buffer_size, architect, gamma, tau, batch_size, device)
            do_replay(b_replay_buffer, min_buffer_size, builder,   gamma, tau, batch_size, device)

            state_with_goal = new_state
        b_replay = Transition(s=old_message, a=action, r=reward, next_s=message, t=done)
        b_replay_buffer.add(b_replay)
        eps = max(eps * eps_decay, eps_end)

        print(f"Episode: {i}, Eps: {round(eps,3)}, Last Reward (env): {reward}")
