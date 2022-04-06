import gym
import gym_builderarch
import sys
import torch
import torch.nn.functional as F
from collections import namedtuple
from agents import Architect, Builder

difficulty = sys.argv[1]
replace = bool(sys.argv[2])

env = gym.make('BuilderArch-v1')
env.reset(difficulty = difficulty)

device = torch.device('cpu')

max_catalog_size = 0
grouped_actions = env.action_space
num_actions = grouped_actions[0].n * grouped_actions[1].n
num_states  = env.size
architect   = Architect(num_states,                    num_actions, 2, device, 'dense', training = False, max_catalog_size = max_catalog_size, grouped = True)
builder     = Builder(int(architect.dqn._num_actions), num_actions, 1, device, 'smalldense', training = False, max_catalog_size = max_catalog_size, grouped = True)


def iterate(example_index, architect, builder, env, device, difficulty="normal"):
    GlobalTransition = namedtuple("GlobalTransition", ["s", "m", "a", "r", "t"])
    env.reset(difficulty = difficulty)
    # Get the i:th example and set it as goal
    ex = env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[example_index][1]
    env.set_goal(ex)
    print("------GOAL------")
    env.render_state(env.goal)
    print("----------------")
    
    done = False
    episode_history = []
    steps = 0
    while not done:
        state = env.get_state()[None,:]
        env.render()
        steps += 1
        print("---------")
        print(str([x for x in range(env.size[0])]) + " Vert")
        print(str([x+env.size[0] for x in range(env.size[0])]) + " Hori")
        action = torch.tensor(int(input('Take action ')))
        action_one_hot = F.one_hot(action,num_classes = builder.num_actions).float()
        action_one_hot = action_one_hot[None,:]
        # Env: Take action
        (new_state, reward, done, success) = builder.build(action, env)
        
        print(f"Message: Action: {action}, Reward: {reward}")

        new_state = new_state[None,:]
        nonterminal_to_buffer = not done or steps == 100
        episode_history.append(GlobalTransition(s=state, m = action, a = action, r = reward, t = nonterminal_to_buffer))
    episode_history.append(GlobalTransition(s=new_state, m = None, a = None, r = None, t = None))

    architect.append_buffer(episode_history)
    builder.append_buffer(episode_history)
    print(f"\n--RESULT, GOAL-- in {env.steps} steps (ex:{example_index})")
    env.render_state_with_goal(env.state, env.goal)
    print("----------------\n\n")

ex = 0
trials = 0
location = f"./gym_builderarch/envs/supervised_play/{difficulty}{env.size[0]}.replay"

print(location)
print(f"Replacing: {replace}")
while ex != -1:
    try:
        iterate(ex, architect, builder, env, device, difficulty = difficulty)
        ex = int(input('What example do you want to do now? (-1 to quit) '))
        trials += 1
    except Exception as e:
        print(e)
        break
architect.replay_buffer.store(location, trials, replace = replace)