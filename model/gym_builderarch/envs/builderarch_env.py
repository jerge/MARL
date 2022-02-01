import math
from typing import Optional

import gym
from gym import spaces, logger
from gym.utils import seeding
import torch
import copy
class BuilderArchEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    def __init__(self):
        self.goal = None
        self.size = (7,7)
        self.loc = 0
        self.action_space = spaces.Discrete(4) # h v l r
        self.no_action_punishment = -100
        self.state = None

    def set_goal(self, goal):
        assert self.state.shape == goal.shape
        self.goal = goal

    # TODO: Reasonable?
    def is_done(self):
        assert self.goal != None, "No goal has been set and the environment can thus not be done"
        return torch.sum(torch.roll(self.state,-self.loc,1)) > torch.sum(self.goal) or torch.sum(torch.roll(self.state,-self.loc,1)-self.goal)==0

    def step(self, action):
        prev_state = copy.deepcopy(self.state)
        allowed = self.take_action(action)
        ob = self.state
        reward = self.get_reward(ob,prev_state) if allowed else self.no_action_punishment
        done = self.is_done()
        return ob, reward, done, {}

    # TODO: Maybe have location as a circular variable i.e. state[0] = state[0] % self.size[1]
    def reset(self):
        # MAYBE WANT TO USE RANDOM SEED AT ONE POINT
        self.state = torch.zeros(self.size)
        self.loc = 0
        return self.state

    def render(self, mode='human'):
        print(torch.roll(self.state.long(),-self.loc,1))
        return False

    
    # Returns wether the action was allowed and mutates the state if it was allowed
    def take_action(self, action):
        action = int(action)
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        # Returns the top location of every column
        top_locations = [x.shape[0] if torch.max(x) == 0 else torch.argmax(x) for x in self.state.T]

        # Vertical block
        if action == 0: # v
            loc = 0
            top = top_locations[loc]

            if top <= 1:
                return False
            self.state[top-2,loc] = 1
            self.state[top-1,loc] = 1
        # Horisontal block
        elif action == 1: # h
            loc = 0
            if loc >= self.size[1]-2:
                return False
            top = min(top_locations[loc],top_locations[loc+1 % self.size[1]])

            if not (top > 0):
                return False
            self.state[top-1, loc] = 1
            self.state[top-1, loc+1 % self.size[1]] = 1
        elif action == 2: # l
            self.loc -= 1
            self.state = torch.roll(self.state,-1,1)
        elif action == 3: # r
            self.loc += 1
            self.state = torch.roll(self.state,1,1)
        return True

    def get_reward(self, state, prev_state):
        """ Reward is given for XY. """
        if self.goal is None:
            return 0
        return torch.sum(self.goal*(state[1]-prev_state[1]))