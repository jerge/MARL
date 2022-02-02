import math
from typing import Optional

import gym
from gym import spaces, logger
from gym.utils import seeding
import torch
import copy
import random

class BuilderArchEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    def __init__(self):
        self.goal = None
        self.size = (7,7)
        self.loc = 0
        self.action_space = spaces.Discrete(4) # h v l r
        self.no_action_punishment = -100
        self.state = None
        self.steps = 0
        self.max_steps = 100

    def set_goal(self, goal):
        assert self.state.shape == goal.shape
        self.goal = goal

    # TODO: Reasonable?
    def is_done(self):
        assert self.goal != None, "No goal has been set and the environment can thus not be done"
        return torch.sum(self.state) >= torch.sum(self.goal)  \
                or torch.sum(self.state - self.goal) ==0  \
                or self.steps >= self.max_steps

    def step(self, action):
        self.steps += 1
        prev_loc = self.loc
        prev_state = copy.deepcopy(self.state)

        allowed = self.take_action(action)

        reward = self.get_reward(prev_state,prev_loc) if allowed else self.no_action_punishment
        #print(reward)
        done = self.is_done()
        ob = torch.roll(self.state,self.loc,1)
        return ob, reward, done, {}

    # TODO: Maybe have location as a circular variable i.e. state[0] = state[0] % self.size[1]
    def reset(self):
        # MAYBE WANT TO USE RANDOM SEED AT ONE POINT
        self.state = torch.zeros(self.size)
        self.loc = 0
        self.steps = 0
        ex = random.choice(get_easy_examples7())
        self.set_goal(ex)
        return self.state

    def render(self, mode='human'):
        print(self.state.long())
        return False

    
    # Returns wether the action was allowed and mutates the state if it was allowed
    def take_action(self, action):
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
            if loc >= self.size[1]-2:
                return False
            top = min(top_locations[loc],top_locations[(loc+1) % self.size[1]])

            if not (top > 0):
                return False
            self.state[top-1, loc] = 1
            self.state[top-1, (loc+1) % self.size[1]] = 1
        elif action == 2: # l
            self.loc = (self.loc - 1) % self.size[1]
            self.state = self.state
        elif action == 3: # r
            self.loc = (1 + self.loc) % self.size[1]
            self.state = self.state
        return True

    def get_reward(self, prev_state, prev_loc):
        """ Reward is given for XY. """
        if self.goal is None:
            return 0
        diff_state = (self.state - prev_state)
        return torch.sum(self.goal * (diff_state) * 2- diff_state)

# TODO: PLEASE FIX
def get_examples7():
    examples7 = []
    size = (7,7)

    l1 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0],
                        [1,1,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(l1)

    l2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,1,1,1,0,0]],dtype=torch.int32)
    examples7.append(l2)

    l3 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(l3)

    l4 =  torch.tensor([[0,0,0,0,0,0,0,],
                        [0,0,0,0,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,1,1,1,]],dtype=torch.int32)
    examples7.append(l4)

    l1r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [1,1,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(l1r)

    l2r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,1,1,1,1,0,0]],dtype=torch.int32)
    examples7.append(l2r)

    l3r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(l3r)

    l4r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,1,1,1,1]],dtype=torch.int32)
    examples7.append(l4r)


    u4 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,1,1,1]],dtype=torch.int32)
    examples7.append(u4)

    u3 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(u3)

    u2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,1,1,1,0,0]],dtype=torch.int32)
    examples7.append(u2)

    u1 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,1,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(u1)

    pi1 = copy.deepcopy(u1).T.T
    pi2 = copy.deepcopy(u2).T.T
    pi3 = copy.deepcopy(u3).T.T
    pi4 = copy.deepcopy(u4).T.T
    examples7.append(pi1)
    examples7.append(pi2)
    examples7.append(pi3)
    examples7.append(pi4)
    # def one_zero(loc):
    #     o = torch.ones(size,dtype=torch.int32)
    #     o[loc] = 0
    #     return o
    # locs = [(x,y) for y in range(7) for x in range(7)]
    # examples7 = examples7 + list(map(one_zero, locs))

    H1 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,0,0,1,0,0,0]],dtype=torch.int32)
    examples7.append(H1)

    H2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,1,1,1,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,0,0,1,0,0]],dtype=torch.int32)
    examples7.append(H2)

    H3 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,1,1,1,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,0,0,1,0]],dtype=torch.int32)
    examples7.append(H3)

    H4 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,1,1,1],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,0,0,1]],dtype=torch.int32)
    examples7.append(H4)

    Ht3 = copy.deepcopy(H4).T
    examples7.append(Ht3)
    Ht2 = torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,1,1,1,1,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,1,1,1,1,1,0]], dtype=torch.int32)
    examples7.append(Ht2)

    Ht1 = torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,1,1,1,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [1,1,1,1,1,0,0]], dtype=torch.int32)
    examples7.append(Ht1)

    t2 = torch.tensor([[0,0,0,0,0,0,0],
                       [0,1,1,1,1,1,1],
                       [0,0,1,1,1,1,0],
                       [0,0,0,1,1,0,0],
                       [0,0,0,1,1,0,0],
                       [0,0,0,1,1,0,0],
                       [0,0,0,1,1,0,0]],dtype=torch.int32)
    examples7.append(t2)

    t1 =  torch.tensor([[0,0,0,0,0,0,0],
                       [1,1,1,1,1,1,0],
                       [0,1,1,1,1,0,0],
                       [0,0,1,1,0,0,0],
                       [0,0,1,1,0,0,0],
                       [0,0,1,1,0,0,0],
                       [0,0,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(t1)

    it = copy.deepcopy(t2).T.T.index_select(0,torch.tensor([0,6,5,4,3,2,1]))
    examples7.append(it)

    pyr =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,1,0,0,1],
                        [0,0,1,1,0,0,1],
                        [0,1,1,1,1,0,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(pyr)

    pyr2 =  torch.tensor([[0,0,0,0,0,1,1],
                        [1,1,0,0,0,0,1],
                        [1,0,0,0,0,0,1],
                        [1,0,1,1,0,0,1],
                        [1,0,1,1,0,0,1],
                        [1,1,1,1,1,0,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(pyr2)

    tri =  torch.tensor([[0,0,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(tri)

    trix =  torch.tensor([[1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(trix)

    tri2 = copy.deepcopy(trix)
    tri2 = torch.flip(tri2, [1,0])
    examples7.append(tri2)


    blo =  torch.tensor([[0,0,0,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(blo)


    pin3 =  torch.tensor([[0,0,0,1,1,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,1],
                        [1,1,1,1,0,0,1],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(pin3)

    shapa =  torch.tensor([[0,0,0,1,1,0,0],
                        [0,0,1,1,0,0,0],
                        [0,0,1,1,0,0,0],
                        [0,1,1,1,0,0,1],
                        [0,1,1,1,0,0,1],
                        [0,1,1,1,1,1,1],
                        [0,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(shapa)


    shapab =  torch.tensor([[0,0,0,1,1,0,0],
                        [0,0,1,1,0,0,0],
                        [0,0,1,1,0,0,0],
                        [0,0,1,1,0,0,1],
                        [1,1,1,1,0,0,1],
                        [1,0,1,1,1,1,1],
                        [1,0,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(shapab)

    mill =  torch.tensor([[0,0,1,1,0,0,0],
                          [0,0,0,1,0,1,1],
                          [0,0,0,1,1,1,0],
                          [0,0,0,1,1,0,0],
                          [1,1,1,1,0,0,0],
                          [1,0,0,1,0,0,0],
                          [1,0,0,1,1,1,0]], dtype=torch.int32)
    examples7.append(mill)


    arch =  torch.tensor([[0,0,1,1,0,0,0],
                          [0,1,1,0,1,1,0],
                          [0,1,0,0,0,1,0],
                          [0,1,0,0,0,1,0],
                          [1,1,0,0,0,1,1],
                          [1,0,0,0,0,0,1],
                          [1,0,0,0,0,0,1]],dtype=torch.int32)
    examples7.append(arch)

    arch2 = torch.tensor([[0,0,0,1,1,0,0],
                        [0,1,1,0,1,1,0],
                        [0,1,0,0,0,1,0],
                        [0,1,0,0,0,1,0],
                        [1,1,0,0,0,1,1],
                        [1,0,0,0,0,0,1],
                        [1,0,0,0,0,0,1]],dtype=torch.int32)
    examples7.append(arch2)

    archb =  torch.tensor([[0,0,1,1,0,0,0],
                        [0,1,1,0,1,1,0],
                        [0,1,0,0,0,1,0],
                        [0,1,0,0,0,1,0],
                        [1,1,0,0,0,1,1],
                        [1,0,0,1,1,0,1],
                        [1,0,0,1,1,0,1]],dtype=torch.int32)
    examples7.append(archb)

    hinv =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,0,1,0,0],
                        [0,0,1,0,1,0,0]],dtype=torch.int32)
    examples7.append(hinv)


    hbas =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,0,1,0,0],
                        [0,0,1,0,1,0,0]],dtype=torch.int32)
    examples7.append(hbas)


    brig =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,0,1,0,1,0,1],
                        [1,0,1,1,1,0,1]],dtype=torch.int32)
    examples7.append(brig)

    brig =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,0,1,0,1,0,1],
                        [1,0,1,1,1,0,1]],dtype=torch.int32)
    examples7.append(brig)
    return examples7

def get_easy_examples7():
    return get_examples7()[:16]