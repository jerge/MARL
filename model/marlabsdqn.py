import torch
import torch.nn.functional as F
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple, deque
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
import numpy as np
from sleeping import generate_abstraction, find_bad_abstractions
from dreaming import dream
import pandas as pd
import os

# Tests if the current dqn can solve all n_examples with epsilon 0
def test_examples(n_examples, architect, builder, env, device, difficulty="normal", eps = 0, pretty_test=False, evaluation = False):
    successful = True
    if pretty_test:
        action_list = ['Vert','Hori','Left','Right']
        a_catalog_names = action_list + [",".join([action_list[item][0] for item in itemlist]) for itemlist in [items.tolist() for items in architect.catalog]]
        b_catalog_names = action_list + [",".join([action_list[item][0] for item in itemlist]) for itemlist in [items.tolist() for items in builder.catalog]]

    #r_threshold = 0
    rewards = []
    for i in range(n_examples):
        env.reset(difficulty = difficulty)
        # Get the i:th example and set it as goal
        ex = env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1]
        env.set_goal(ex)
        if pretty_test:
            print("------GOAL------")
            env.render_state(env.goal)
            print("----------------")
        done = False
        ep_reward = 0
        while not done:
            state = env.get_state()[None,:]
            if pretty_test:
                env.render()
            # A: Generate message from state
            q_a, message_one_hot = calc_q_and_take_action(architect, state, eps, device)
            message = torch.argmax(message_one_hot[:architect.available_actions()])
            message_one_hot = message_one_hot[None,:]
            
            # B: Generate action from message
            q_b, action_one_hot = calc_q_and_take_action(builder, message_one_hot, eps, device, debug=False, symbolic = True) # eps
            action = message#torch.argmax(action_one_hot[:builder.available_actions()])
            action_one_hot = message_one_hot#action_one_hot[None,:]
            # Env: Take action
            (new_state, reward, done, success) = builder.build(action, env)
            ep_reward += reward
            if pretty_test:
                print(list(zip(a_catalog_names,[round(x,3) for x in q_a.tolist()[0]])))
                print(list(zip(b_catalog_names,[round(x,3) for x in q_b.tolist()[0]])))
                print(f"Message: {a_catalog_names[message]}, Action: {b_catalog_names[action]}, Reward: {reward}, New loc: {env.loc}")

            new_state = new_state[None,:]
        rewards.append(ep_reward)
        if not success:
            successful = False
        if not evaluation:
            print(f"\n--RESULT, GOAL-- in {env.steps} steps with {eps*100}% randomness")
            env.render_state_with_goal(env.state, env.goal)
            print("----------------\n\n")
        if not success and not evaluation:
            print(f"Could not solve example #{i}")
            return (False, reward)
    if successful:
        print(f"Solved all {n_examples} examples.")
    return (successful,rewards)

def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        r = random.randint(0,q_values.shape[1]-1)
        return r
    return torch.argmax(q_values)

def calc_q_and_take_action(agent, state, eps, device, debug = False, symbolic = False):
    if symbolic:
        sym = agent.use_symbol(state)
        if sym != None:
            return {}, F.one_hot(torch.tensor(sym), num_classes = agent.max_actions()).float()

    q_online_curr = agent.dqn.online_model(state.to(device=device)).cpu()
    # Crop the maximum to not allow actions outside the current catalog
    action_i = torch.tensor(eps_greedy_policy(q_online_curr[0][:agent.available_actions()][None,:], eps))
    if debug:
        print(q_online_curr)
    return q_online_curr, F.one_hot(action_i,num_classes = q_online_curr.shape[1]).float()

def calculate_q_targets(q1_batch, r_batch, nonterminal_batch, gamma=.99):
    Y = r_batch + nonterminal_batch.long() * gamma * (torch.max(q1_batch,dim=1)[0])
    return Y

def sample_batch_and_calculate_loss(dqn, replay_buffer, batch_size, gamma, device, sample_latest = False):
    if sample_latest:
        curr_state, curr_action, reward, next_state, nonterminal = replay_buffer.sample_latest(batch_size)
    else:
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

def wake(env, architect, builder, episode_buffer, eps, eps_end, tau, batch_size, min_buffer_size, device, n_examples, difficulty):
    GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])
    state = env.reset(n=n_examples, difficulty=difficulty)[None,:] # Initial state with added batch size

    (ep_reward, steps, done) = (0, 0, False)
    episode_history = [] # list of all 'GlobalTransition's in the current episode
    while not done:
        steps += 1
        state = env.get_state()[None,:]
        # A: Generate message from state
        if architect.training:
            with torch.no_grad():
                _, message_one_hot = calc_q_and_take_action(architect, state, eps, device)
        else:
            with torch.no_grad():
                _, message_one_hot = calc_q_and_take_action(architect, state, eps_end, device)
        message = torch.argmax(message_one_hot)#[:architect.num_actions + len(architect.catalog)])
        action_one_hot = message_one_hot
        message_one_hot = message_one_hot[None,:]
        
        # B: Generate action from message
        # if builder.training:
        #     with torch.no_grad():
        #         _, action_one_hot = calc_q_and_take_action(builder, message_one_hot, eps, device, symbolic = True) # eps
        # else:
        #     with torch.no_grad():
        #         _, action_one_hot = calc_q_and_take_action(builder, message_one_hot, eps_end, device, symbolic = True)
        action = torch.argmax(action_one_hot)#[:builder.num_actions + len(builder.catalog)])
        action_one_hot = action_one_hot[None,:]

        # Env: Take action
        (new_state, reward, done, success) = builder.build(action, env)
        new_state = new_state[None,:]

        # Save transition
        nonterminal_to_buffer = not done or steps == 100
        episode_history.append(GlobalTransition(s=state, m = message, a = action, r = reward, t = nonterminal_to_buffer))

        ep_reward += reward
    # Extra transition for architect
    episode_history.append(GlobalTransition(s=new_state, m = None, a = None, r = None, t = None))
    
    # Add the episode to the replay buffers
    # Do not let the architect's training depend on a training builder
    if not builder.training:
        architect.append_buffer(episode_history)
    builder.append_buffer(episode_history)
    
    # Add the episode, if successful, to the episode buffer for later abstraction
    if success > 0:
        episode_buffer.append(episode_history)
    
    do_replay(architect.replay_buffer, min_buffer_size, architect.dqn, architect.gamma, tau, batch_size, device)
    do_replay(builder.replay_buffer,   min_buffer_size, builder.dqn,   builder.gamma,   tau, batch_size, device)

    return (steps, ep_reward)

def train_loop(env, architect, builder, n_episodes, 
                device, n_examples, difficulty,
                batch_size = 512, lim = 9, df_path = ".", n_plot_examples = 1):
    min_buffer_size = 100
    (eps, eps_decay, eps_end) = (0.99, 0.9995, 0.03)
    (last_lim_change, init_lim) = (0, lim)
    tau = 200 # Frequency of target network updates
    R_avg = 0 # Running average of episodic rewards (total reward, disregarding discount factor)
    tot_steps = 0
    (trial,cleared_before) = (False,False)
    high_eps_episode = False

    episode_buffer = deque(maxlen=500) # queue of entire episodes
    for i in range(n_episodes):
        if random.randint(0,10) == 0:
            prev_eps = eps
            eps = 0.9
            high_eps_episode = True
        (steps, ep_reward) = wake(env, architect, builder, episode_buffer, 
                                    eps, eps_end, tau, batch_size, min_buffer_size, 
                                    device, n_examples, difficulty)
        if high_eps_episode:
            high_eps_episode = False
            eps = prev_eps
            continue
        eps = max(eps * eps_decay, eps_end)

        p = 1/min(i+1,1000) # The proportion that the current episode should count towards R_avg
        R_avg =  p * ep_reward + (1-p) * R_avg
        t = "a,b" if architect.training and builder.training else "a" if architect.training else "b" if builder.training else "none"
        if i % 50 == 0:
            print('Episode: {:d}, #Ex: {:.0f}, Steps: {: 3d}, Ep Reward (running avg): {:4.0f} ({:.2f}) Eps: {:.3f}, Trainee: {}, Cat: {}, #Symb: {}'.format(
                                                                                    i, n_examples, steps, ep_reward, R_avg, eps, t, 
                                                                                    architect.catalog, len(builder.symbols.keys())))

        # If there has been 1000 steps since last time, switch trainee and do a trial
        if (tot_steps + steps) % 1000 < tot_steps % 1000:
            trial = True
            # architect.training = architect.training != True
            # builder.training   = builder.training   != True
            # if builder.learn_symbol():
            #     print("Learnt symbols:")
            #     print(builder.symbols.items())
            
        tot_steps += steps

        #if R_avg > lim/(lim+1):
        if trial:
            cleared_examples = test_examples(n_examples, architect, builder, env, device, difficulty=difficulty)[0]
            # --- Record current performance ---
            if i < 1000 or i % 1000 < 50 or cleared_examples:
                rewards = []
                for _ in range(3):
                    with torch.no_grad():
                        # NOTE n = 7
                        rewards = rewards + test_examples(n_plot_examples, architect, builder, env, device, difficulty = difficulty, eps = eps, evaluation = True)[1]
                if os.path.isfile(f'{df_path}/rewards{n_examples}.csv'):
                    df = pd.read_csv(f'{df_path}/rewards{n_examples}.csv')
                else:
                    df = pd.DataFrame(columns = ["num_episodes","R_avg_tot", "R_avg_curr"])
                df = df.append(pd.DataFrame([(i,np.mean(rewards),R_avg)], columns = ["num_episodes","R_avg_tot", "R_avg_curr"]))
                df.to_csv(f'{df_path}/rewards{n_examples}.csv', index=False)
            # ----------------------------------
            
            print("---------------------------------")
            if cleared_examples:
                if cleared_before:
                    return R_avg, False
                cleared_before = True
            print("---------------------------------")
            if random.randint(0,len(architect.catalog)) == 0 and i > 10000:
                abstractions = generate_abstraction(episode_buffer, env.size[0], architect.catalog, grouped = env.grouped)
                if len(abstractions) > 0:
                    abstraction = abstractions[0]
                    architect.increase_catalog(abstraction)
                    builder.increase_catalog(abstraction)
                    dream(architect, builder, abstractions, env, episode_buffer, device)
                else:
                    actions = architect.replay_buffer.sample_latest(batch_size=1000)[1].cpu().detach().numpy()
                    bad_abstractions = find_bad_abstractions(actions, eps, architect.num_std_blocks, architect.num_actions, len(architect.catalog))
                    if len(bad_abstractions) > 0:
                        print(f"Removing: Catalog action(s) {bad_abstractions}, due to seldom usage")
                        print(builder.catalog)
                        print(bad_index-builder.num_std_blocks)
                        [architect.catalog.pop(bad_index-architect.num_std_blocks) for bad_index in bad_abstractions]
                        [builder.catalog.pop(bad_index-builder.num_std_blocks) for bad_index in bad_abstractions]
                

            lim += 1
            trial = False
            last_lim_change = i
        if i - last_lim_change > 1000:
            lim = init_lim
            last_lim_change = i
    return R_avg, True
