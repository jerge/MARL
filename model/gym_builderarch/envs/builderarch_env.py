import math
from typing import Optional

import gym
from gym import spaces, logger
from gym.utils import seeding
import torch
from collections import deque

import copy
import random

class BuilderArchEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    def __init__(self):
        self.goal = None
        self.size = (5,5)
        self.loc = 0
        self.action_space = spaces.Discrete(4) # v h l r
        self.invalid_action_punishment = torch.tensor(0.)
        self.state = None
        self.steps = 0
        self.max_steps = 100
        self.prev_actions = deque(maxlen=int(self.size[0]))

    def set_goal(self, goal):
        assert self.state.shape == goal.shape, f"state: {self.state.shape}, goal: {goal.shape}"
        self.goal = goal

    def is_done(self):
        assert self.goal != None, "No goal has been set and the environment can thus not be done"
        # If more blocks have been placed than the goal requires,
        # or if the goal is reached
        # or if the self.max_steps amount of steps has been reached
        return torch.sum(self.state) >= torch.sum(self.goal)  \
                or torch.sum(self.state - self.goal) == 0  \
                or self.steps >= self.max_steps
    
    def step(self, action):
        self.steps += 1
        prev_state = copy.deepcopy(self.state)
        allowed = self.take_action(action)

        done = self.is_done()
        
        self.prev_actions.append(action)
        # (reward, success) = self.get_reward(done) if allowed else (self.invalid_action_punishment, False)
        (reward, success) = self.get_reward_2(prev_state, done) if allowed else (self.invalid_action_punishment, False)

        ob = self.get_state()
        return ob, reward, done, success

    def reset(self, n=1, difficulty = "normal"):
        self.state = torch.zeros(self.size)
        self.loc = 0
        self.steps = 0
        # Examples are of form (name,grid)
        if random.randint(0,3) == 0:
            ex = self.get_examples(filename=f"{difficulty}{self.size[0]}.squares")[n][1]
        else:
            ex = random.choice(self.get_examples(filename=f"{difficulty}{self.size[0]}.squares")[:n])[1]
        self.set_goal(ex)
        return self.get_state()

    def render(self, mode='human'):
        print(" "*(3*self.loc+1) + "\N{WHITE DOWN POINTING BACKHAND INDEX}")
        for row in self.state.long().tolist():
            print(str(row).replace('0',"."))
        return False
    
    def render_state(self, state):
        print(" "*(3*self.loc+1) + "\N{WHITE DOWN POINTING BACKHAND INDEX}")
        for row in state.long().tolist():
            print(str(row).replace('0',"."))
        return False

    def get_state(self):
        #ob = torch.stack((torch.roll(self.goal,-self.loc,1), torch.roll(self.state,-self.loc,1)))
        st = copy.deepcopy(self.state)
        st[0][self.loc] -= 1
        ob = torch.stack((self.goal, st))
        return ob
    
    # Returns true if the action was allowed and mutates the state if it was allowed
    def take_action(self, action):
        if len(action.size()) != 0:
            return self.take_actions(action)
        action = int(action)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        # Returns the top location of every column
        top_locations = [x.shape[0] if torch.max(x) == 0 else torch.argmax(x) for x in self.state.T]

        # Vertical block
        if action == 0: # v
            loc = self.loc
            top = top_locations[loc]

            if top <= 1:
                return False
            self.state[top-2,loc] = 1
            self.state[top-1,loc] = 1
        # Horisontal block
        elif action == 1: # h
            loc = self.loc
            top = min(top_locations[loc],top_locations[(loc+1) % self.size[1]])
            if not (top > 0):
                return False
            self.state[top-1, loc] = 1
            #print(self.state)
            self.state[top-1, (loc+1) % self.size[1]] = 1
        elif action == 2: # l
            self.loc = (self.loc - 1) % self.size[1]
            #self.state = self.state
        elif action == 3: # r
            self.loc = (1 + self.loc) % self.size[1]
            #self.state = self.state
        return True

    # Returns true if any of the actions were allowed and mutates the state
    def take_actions(self, action_list):
        return any([self.take_action(action) for action in action_list])

    # Binary reward function that accounts for n_steps by 0.99^steps
    # Also returns if the construct was perfect
    def get_reward(self, done):
        if self.goal is None:
            return torch.tensor(0.)
        if done:
            if torch.equal(self.goal, self.state.long()):
                return (torch.tensor(1.) * (0.90**self.steps), True)
        return (torch.tensor(0.), False)
    
    # Binary reward but also gives intermediate reward 0.1 * 1/(n_blocks) every time a block is
    # placed in a potentially correct spot
    def get_reward_2(self, prev_state, done):
        (reward, success) = self.get_reward(done)
        if torch.sum(self.state-prev_state) == 2:
            reward += 0.1 * (1/(torch.sum(self.goal)//2)) * (0.90 ** self.steps)
        return (reward, success)

    def get_reward_3(self, done):
        (reward, success) = self.get_reward(done)
        return (reward * (torch.sum(self.goal)//2), success)

    
    def get_examples(self, filename=f"generated7.squares"):
        import os
        read_file = open(os.path.join(os.path.dirname(__file__), filename),"r").read()
        import re
        examples = []
        for example in re.findall("(\w+)--((\n\[(\w|, )+\])+)", read_file):
            name = example[0]
            shape = example[1].replace("[", "").replace("]","").strip()
            line_strings = shape.split("\n")
            matrix = torch.tensor([[int(element) for element in line_string.split(",")] for line_string in line_strings],dtype=torch.int64)
            examples.append((name, matrix))
        return examples

    
        
