import gym
import gym_builderarch
import numpy as np
import examples as ex

env = gym.make('BuilderArch-v1')
ex1 = ex.get_examples7()[1]
print(ex1)
env.reset()
env.set_goal(ex1)

env.render()
s,r,d,_ = env.step(env.action_space.sample())
print(r)
env.render()
s,r,d,_ = env.step(env.action_space.sample())
print(r)
env.render()
