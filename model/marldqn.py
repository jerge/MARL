import torch
import torch.nn.functional as F
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
import numpy as np

# Tests if the current dqn can solve all n_examples with epsilon 0
def test_examples(n_examples, architect, builder, env, device, difficulty="normal"):
    eps = 0
    r_threshold = 0
    for i in range(n_examples):
        env.reset()
        ex = env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1]
        env.set_goal(ex)

        done = False
        while not done:
            state = env.get_state()
            state = state[None,:]

            _, m = calc_q_and_take_action(architect, state, eps, device)
            m = m[None,:]
            _, a = calc_q_and_take_action(builder, m, eps, device)
            a = torch.argmax(a)
            ob, r, done, _ = env.step(a)
        if not r > r_threshold:
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

def train_loop(env, architect, builder, n_episodes, a_replay_buffer, b_replay_buffer, 
                device, n_examples, difficulty, catalog,
                batch_size = 128, training_architect = True, training_builder = True, lim = 9):
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])
    min_buffer_size = 100
    gamma = 0.95
    (eps, eps_decay, eps_end) = (0.99, 0.999, 0.03)
    (last_lim_change, init_lim) = (0, lim)
    trial = False
    tau = 500 # Frequency of target network updates
    R_avg = 0 # Running average of episodic rewards (total reward, disregarding discount factor)
    tot_steps = 0
    for i in range(n_episodes):
        state_with_goal = env.reset(n=n_examples, difficulty=difficulty) # Initial state
        state_with_goal = state_with_goal[None,:]
        message_one_hot = torch.zeros(builder._num_states)

        ep_reward = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            tot_steps += 1
            if tot_steps % 1000 == 0:# and i != 0:
                trial = True
                training_architect = training_architect != True
                training_builder   = training_builder   != True
            # A: Generate message from state_with_goal
            if training_architect:
                _, message_one_hot = calc_q_and_take_action(architect, state_with_goal, eps, device)
            else:
                with torch.no_grad():
                    _, message_one_hot = calc_q_and_take_action(architect, state_with_goal, 0.03, device)
            message = torch.argmax(message_one_hot)
            message_one_hot = message_one_hot[None,:]
            
            #B: Add (s,a,r,s') and train
            if training_builder and steps > 1:
                # TODO: What reward should the builder get
                # TODO: Does the builder need one extra iteration?
                b_reward = int(action == torch.argmax(old_message_one_hot))*3-2
                b_replay = Transition(s=old_message_one_hot, a=action, r=b_reward, next_s=message_one_hot, t=nonterminal_to_buffer)
                b_replay_buffer.add(b_replay)
                do_replay(b_replay_buffer, min_buffer_size, builder,   gamma, tau, batch_size, device)

            # B: Generate action from message
            if training_builder:
                q, action_one_hot = calc_q_and_take_action(builder, message_one_hot, eps, device, debug=False) # eps
            else: 
                with torch.no_grad():
                    _, action_one_hot = calc_q_and_take_action(builder, message_one_hot, 0.03, device)
            action = torch.argmax(action_one_hot)

            # Env: Take action
            new_state, reward, done, _ = env.step(action)
            new_state = new_state[None,:]
            
            nonterminal_to_buffer = not done or steps == 99

            #A: Add (s,a,r,s') and train
            if training_architect:
                a_replay = Transition(s=state_with_goal, a=message, r=reward, next_s=new_state, t=nonterminal_to_buffer)
                a_replay_buffer.add(a_replay)
                do_replay(a_replay_buffer, min_buffer_size, architect, gamma, tau, batch_size, device)

            ep_reward += reward
            state_with_goal = new_state
            old_message_one_hot = message_one_hot
        # Builder extra iteration
        #B: Add (s,a,r,s') and train
        if training_builder and steps >= 1:
            # TODO: What reward should the builder get
            # TODO: Does the builder need one extra iteration?
            b_reward = int(action == torch.argmax(old_message_one_hot))*3-2
            b_replay = Transition(s=old_message_one_hot, a=action, r=b_reward, next_s=message_one_hot, t=nonterminal_to_buffer)
            b_replay_buffer.add(b_replay)
            do_replay(b_replay_buffer, min_buffer_size, builder,   gamma, tau, batch_size, device)

        eps = max(eps * eps_decay, eps_end)

        R_avg =  (1-gamma) * ep_reward + (gamma) * R_avg
        t = "a,b" if training_architect and training_builder else "a" if training_architect else "b" if training_builder else "none"
        print('Episode: {:d}, Ex: {:.0f}, Total Reward (running avg): {:4.0f} ({:.2f}) Epsilon: {:.3f}, Trainee: {}'.format(
                                                                                    i, n_examples, ep_reward, R_avg, eps, t))

        #if R_avg > lim/(lim+1):
        if trial:
            print("---------------------------------")
            if test_examples(n_examples, architect, builder, env, device, difficulty=difficulty):
                return R_avg, False
            lim += 1
            print("---------------------------------")
            trial = False
            last_lim_change = i
        if i - last_lim_change > 1000:
            lim = init_lim
            last_lim_change = i
    return R_avg, True