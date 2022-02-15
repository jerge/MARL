import gym
import gym_builderarch
import numpy as np

env = gym.make('BuilderArch-v1')
#ex1 = ex.get_examples7()[1]
#print(ex1)

env.reset()

# env.reset()
# print(env.goal)
# for i in range(10):
    
#     #env.set_goal(ex1)

#     env.render()
#     a = env.action_space.sample()
#     s,r,d,_ = env.step(a)
#     print(f"a:{a}, r:{r}")
#env.render()
#s,r,d,_ = env.step(env.action_space.sample())
#print(r)
#env.render()
