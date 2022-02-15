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
    print('cuda')
    device = torch.device("cuda")
else:
    print('cpu')
    device = torch.device("cpu")

def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        return random.randint(0,q_values.shape[1]-1)
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps):
    q_online_curr = dqn.online_model(state.to(device=device)).cpu()
    action_i = eps_greedy_policy(q_online_curr, eps) # 
    return q_online_curr, torch.tensor(action_i)

def calculate_q_targets(q1_batch, r_batch, nonterminal_batch, gamma=.99):
    Y = r_batch + nonterminal_batch.long() * gamma * (torch.max(q1_batch,dim=1)[0])
    return Y

def sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma):
    curr_state, curr_action, reward, next_state, nonterminal = replay_buffer.sample_minibatch(batch_size)

    q_online_curr = dqn.online_model(curr_state.to(device=device)).cpu()
    with torch.no_grad():
        q_offline_next = dqn.offline_model(next_state.to(device=device)).cpu()
    
    q_target = calculate_q_targets(q_offline_next, reward, nonterminal, gamma=gamma)
    loss = dqn.calc_loss(q_online_curr, q_target, curr_action)

    return loss

def train_loop(env, architect, builder, n_episodes, architect_replay_buffer, batch_size = 128):
    min_buffer_size = 100
    gamma = 0.95
    tau = 100 # Frequency of architect target network updates
    cnt_updates = 0
    for i in range(n_episodes):
        state = env.reset()
        done = False
        episode_transitions = []
        while not done:
            message = architect.predict(env.goal,state)
            action = builder.act(state, message)
            new_state, reward, done, _ = env.step(action)

            replay = Transition(state, action, message, reward, new_state, done)
            architect_replay_buffer.append(replay)
            episode_transitions.append(replay)

            if architect_replay_buffer.buffer_length > min_buffer_size:
                loss = sample_batch_and_calculate_loss(architect.dqn, replay_buffer, batch_size, gamma)
                architect.dqn.optimizer.zero_grad()
                loss.backward()
                dqn.optimizer.step()

                cnt_updates += 1
                if cnt_updates % tau == 0:
                    print("Update target network")
                    architect.dqn.update_target_network()
        # Evaluates the performance of the builder for the past episode
        reward, actions, states = evaluator.evaluate(builder, episode_transitions)
        builder.update(reward, actions, states)

        total_env_reward = sum([transition[1] for transition in episode_transitions])
        print(f"Episode: {i}, Total Reward (env): {total_env_reward}")
        

        