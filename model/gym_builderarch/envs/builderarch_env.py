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
        self.invalid_action_punishment = -0
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

        done = self.is_done()
        reward = self.get_reward(prev_state,done) if allowed else self.invalid_action_punishment
        #print(reward)
        ob = self.get_state()
        return ob, reward, done, {}

    # TODO: Maybe have location as a circular variable i.e. state[0] = state[0] % self.size[1]
    def reset(self):
        # MAYBE WANT TO USE RANDOM SEED AT ONE POINT
        self.state = torch.zeros(self.size)
        self.loc = 0
        self.steps = 0
        # Examples are of form (name,grid)
        ex = random.choice(get_super_easy())[1] #
        #ex = get_easy_examples7()[4][1]
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
        ob = torch.stack((torch.roll(self.goal,self.loc,1), torch.roll(self.state,self.loc,1)))
        return ob
    
    # Returns true if the action was allowed and mutates the state if it was allowed
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

    def get_reward(self, prev_state, done):
        if self.goal is None:
            return torch.tensor(0.)
        if done:
            if torch.equal(self.goal, self.state.int()):
                return torch.tensor(1.)
            # else:
            #     #print(torch.sum(torch.tensor(self.goal*self.state.int()))/(torch.sum(self.goal))-1)
            #     a = torch.sum(torch.tensor(self.goal.float()*self.state))
            #     #print(a)
            #     b = torch.sum(self.goal.float())
            #     #print(b)
            #     #print(a/b)
            #     #print(a/b)
            #     return a/b -1#(torch.sum(torch.tensor(self.goal.float()*self.state))/(torch.sum(self.goal))).long()-1
        return torch.tensor(0.)
        #torch.sum(self.goal * self.state) / torch.sum(self.goal)
        # diff_state = (self.state - prev_state)
        # goal_diff = self.goal * (diff_state)
        # if torch.sum(goal_diff) == 1:
        #     return torch.tensor(0.5)
        # return torch.sum(goal_diff * 2 - diff_state)
    
    def get_all_examples(self):
        return get_examples7()

# TODO: PLEASE FIX
def get_super_easy():
    examples7 = []
    h1 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,1,0,0,0,0]],dtype=torch.int32)
    examples7.append(("h1",h1))

    h2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(("h2",h2))

    v1 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0]],dtype=torch.int32)
    examples7.append(("v1",v1))
    v2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,1,0,0,0],
                        [0,1,0,1,0,0,0]],dtype=torch.int32)
    examples7.append(("v2",v2))

    q =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0]],dtype=torch.int32)
    examples7.append(("q",q))
    return examples7


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
    examples7.append(("l1",l1))

    l2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,0,0,0,0,0],
                        [0,1,1,1,1,0,0]],dtype=torch.int32)
    examples7.append(("l2",l2))

    l3 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(("l3",l3))

    l4 =  torch.tensor([[0,0,0,0,0,0,0,],
                        [0,0,0,0,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,0,0,0,],
                        [0,0,0,1,1,1,1,]],dtype=torch.int32)
    examples7.append(("l4",l4))

    l1r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [1,1,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(("l1r",l1r))

    l2r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,1,1,1,1,0,0]],dtype=torch.int32)
    examples7.append(("l2r",l2r))

    l3r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0],
                        [0,0,0,0,0,1,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(("l3r",l3r))

    l4r =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,0,0,0,1],
                        [0,0,0,1,1,1,1]],dtype=torch.int32)
    examples7.append(("l4r",l4r))


    u4 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,1,1,1]],dtype=torch.int32)
    examples7.append(("u4",u4))

    u3 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,1,1,1,0]],dtype=torch.int32)
    examples7.append(("u3",u3))

    u2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,1,1,1,0,0]],dtype=torch.int32)
    examples7.append(("u2",u2))

    u1 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,0,0,1,0,0,0],
                        [1,1,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(("u1",u1))

    pi1 = copy.deepcopy(u1).T.T
    pi2 = copy.deepcopy(u2).T.T
    pi3 = copy.deepcopy(u3).T.T
    pi4 = copy.deepcopy(u4).T.T
    examples7.append(("pi1",pi1))
    examples7.append(("pi2",pi2))
    examples7.append(("pi3",pi3))
    examples7.append(("pi4",pi4))
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
    examples7.append(("H1",H1))

    H2 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,1,1,1,0,0],
                        [0,1,0,0,1,0,0],
                        [0,1,0,0,1,0,0]],dtype=torch.int32)
    examples7.append(("H2",H2))

    H3 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,1,1,1,0],
                        [0,0,1,0,0,1,0],
                        [0,0,1,0,0,1,0]],dtype=torch.int32)
    examples7.append(("H3",H3))

    H4 =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,1,1,1],
                        [0,0,0,1,0,0,1],
                        [0,0,0,1,0,0,1]],dtype=torch.int32)
    examples7.append(("H4",H4))

    Ht3 = copy.deepcopy(H4).T
    examples7.append(("Ht3",Ht3))
    Ht2 = torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,1,1,1,1,1,0],
                        [0,0,0,1,0,0,0],
                        [0,0,0,1,0,0,0],
                        [0,1,1,1,1,1,0]], dtype=torch.int32)
    examples7.append(("Ht2",Ht2))

    Ht1 = torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,1,1,1,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [1,1,1,1,1,0,0]], dtype=torch.int32)
    examples7.append(("Ht1",Ht1))

    t2 = torch.tensor([[0,0,0,0,0,0,0],
                       [0,1,1,1,1,1,1],
                       [0,0,1,1,1,1,0],
                       [0,0,0,1,1,0,0],
                       [0,0,0,1,1,0,0],
                       [0,0,0,1,1,0,0],
                       [0,0,0,1,1,0,0]],dtype=torch.int32)
    examples7.append(("t2",t2))

    t1 =  torch.tensor([[0,0,0,0,0,0,0],
                       [1,1,1,1,1,1,0],
                       [0,1,1,1,1,0,0],
                       [0,0,1,1,0,0,0],
                       [0,0,1,1,0,0,0],
                       [0,0,1,1,0,0,0],
                       [0,0,1,1,0,0,0]],dtype=torch.int32)
    examples7.append(("t1",t1))

    it = copy.deepcopy(t2).T.T.index_select(0,torch.tensor([0,6,5,4,3,2,1]))
    examples7.append(("it",it))

    pyr =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,1,1,0,0,1],
                        [0,0,1,1,0,0,1],
                        [0,1,1,1,1,0,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("pyr",pyr))

    pyr2 =  torch.tensor([[0,0,0,0,0,1,1],
                        [1,1,0,0,0,0,1],
                        [1,0,0,0,0,0,1],
                        [1,0,1,1,0,0,1],
                        [1,0,1,1,0,0,1],
                        [1,1,1,1,1,0,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("pyr2",pyr2))

    tri =  torch.tensor([[0,0,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("tri",tri))

    trix =  torch.tensor([[1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,0,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("trix",trix))

    tri2 = copy.deepcopy(trix)
    tri2 = torch.flip(tri2, [1,0])
    examples7.append(("tri2",tri2))


    blo =  torch.tensor([[0,0,0,0,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("blo",blo))


    pin3 =  torch.tensor([[0,0,0,1,1,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,0],
                        [1,1,1,1,0,0,1],
                        [1,1,1,1,0,0,1],
                        [1,1,1,1,1,1,1],
                        [1,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("pin3",pin3))

    shapa =  torch.tensor([[0,0,0,1,1,0,0],
                        [0,0,1,1,0,0,0],
                        [0,0,1,1,0,0,0],
                        [0,1,1,1,0,0,1],
                        [0,1,1,1,0,0,1],
                        [0,1,1,1,1,1,1],
                        [0,1,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("shapa",shapa))


    shapab =  torch.tensor([[0,0,0,1,1,0,0],
                        [0,0,1,1,0,0,0],
                        [0,0,1,1,0,0,0],
                        [0,0,1,1,0,0,1],
                        [1,1,1,1,0,0,1],
                        [1,0,1,1,1,1,1],
                        [1,0,1,1,1,1,1]], dtype=torch.int32)
    examples7.append(("shapab",shapab))

    mill =  torch.tensor([[0,0,1,1,0,0,0],
                          [0,0,0,1,0,1,1],
                          [0,0,0,1,1,1,0],
                          [0,0,0,1,1,0,0],
                          [1,1,1,1,0,0,0],
                          [1,0,0,1,0,0,0],
                          [1,0,0,1,1,1,0]], dtype=torch.int32)
    examples7.append(("mill",mill))


    arch =  torch.tensor([[0,0,1,1,0,0,0],
                          [0,1,1,0,1,1,0],
                          [0,1,0,0,0,1,0],
                          [0,1,0,0,0,1,0],
                          [1,1,0,0,0,1,1],
                          [1,0,0,0,0,0,1],
                          [1,0,0,0,0,0,1]],dtype=torch.int32)
    examples7.append(("arch",arch))

    arch2 = torch.tensor([[0,0,0,1,1,0,0],
                        [0,1,1,0,1,1,0],
                        [0,1,0,0,0,1,0],
                        [0,1,0,0,0,1,0],
                        [1,1,0,0,0,1,1],
                        [1,0,0,0,0,0,1],
                        [1,0,0,0,0,0,1]],dtype=torch.int32)
    examples7.append(("arch2",arch2))

    archb =  torch.tensor([[0,0,1,1,0,0,0],
                        [0,1,1,0,1,1,0],
                        [0,1,0,0,0,1,0],
                        [0,1,0,0,0,1,0],
                        [1,1,0,0,0,1,1],
                        [1,0,0,1,1,0,1],
                        [1,0,0,1,1,0,1]],dtype=torch.int32)
    examples7.append(("archb",archb))

    hinv =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,0,0,1,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,0,1,0,0],
                        [0,0,1,0,1,0,0]],dtype=torch.int32)
    examples7.append(("hinv",hinv))


    hbas =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,0,0,0,0],
                        [0,0,1,1,1,0,0],
                        [0,0,1,0,1,0,0],
                        [0,0,1,0,1,0,0]],dtype=torch.int32)
    examples7.append(("hbas",hbas))


    brig =  torch.tensor([[0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1],
                        [1,0,1,0,1,0,1],
                        [1,0,1,1,1,0,1]],dtype=torch.int32)
    examples7.append(("brig",brig))
    return examples7

def get_easy_examples7():
    return get_examples7()[:16]
