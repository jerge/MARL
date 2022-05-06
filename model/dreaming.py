import torch
import marlabsdqn

# Returns if sub_list is a part of full_list
def is_true_sublist(full_list, sub_list):
    subset = set(sub_list)
    if [x for x in full_list if x in subset] == sub_list:
        return True, [i for i,x in enumerate(full_list) if x in subset]
    return False, 0

def dream(architect, builder, new_library, env, history, device):
    if len(new_library) == 0:
        return
    new_samples = []
    for epoch in history:
        actions = [transition.a for transition in epoch]
        for catalog_action in new_library:
            is_sub, indices = is_true_sublist(actions, catalog_action)
            if is_sub:
                new_actions = actions[:indices[0]] + catalog_action + actions[indices[-1]:]
                new_samples.append(new_sample_from_actions(env, new_actions))
    if len(new_samples) == 0:
        return
    for episode_history in new_samples:
        architect.append_buffer(episode_history)
        builder.append_buffer(episode_history)
    print(f'Training the agents on {len(new_samples)} new samples using the new catalog actions "{new_library}"')
    reflect(architect, len(new_samples), device)
    reflect(builder, len(new_samples), device)

def reflect(agent, batch_size, device):
    loss = marlabsdqn.sample_batch_and_calculate_loss(agent.dqn, agent.replay_buffer, batch_size, agent.gamma, device, sample_latest = True)
    dqn.optimizer.zero_grad()
    loss.backward()
    dqn.optimizer.step()

    dqn.num_online_updates += 1
    if dqn.num_online_updates % tau == 0:
        print("Update target network")
        dqn.update_target_network()
                
def new_sample_from_actions(env, new_actions):
    episode_history = []
    env.reset()
    env.set_goal(epoch[0].s[0])
    steps = 0
    for action in new_actions:
        steps += 1
        state = env.get_state()[None,:]

        ob, reward, done, success = env.step(action)
        
        nonterminal_to_buffer = not done or steps == 100
        
        message = aciton
        episode_history.append(GlobalTransition(s=state, m = message, a = action, r = reward, t = nonterminal_to_buffer))
    episode_history.append(GlobalTransition(s=new_state, m = None, a = None, r = None, t = None))
    return episode_history



        