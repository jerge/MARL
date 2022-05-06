import pandas as pd
import collections
import csv
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import os
from torch import tensor

def get_solution(name):
    df = pd.read_csv(f)
    index = df.loc[df['0'] == name]
    return df.at['index', 1]

def get_sublist(full_list, sublist):
    sublist = set(sublist)
    return [x for x in full_list if x in sublist]

# Returns all common sequences in x and y of the largest length
def findCS(x,y):
    parts = [x[a:b] for a, b in combinations(range(len(x) + 1), r = 2)]
    common = [get_sublist(y,part) for part in parts] # Returns the part if it is in y
    # NOTE: return 'common' if you want all common sequences
    if len(common) == 0:
        return []
    max_len = max([len(c) for c in common])
    return set([tuple(c) for c in common if len(c) == max_len])

def decide_abstraction(lcs,n):
    # Currently just taking n most common words
    print(lcs)
    lengths = [word.count('0') + word.count('1') for (word,count) in lcs.most_common()]
    counts = [count for (word,count) in lcs.most_common()]
    print(lengths)
    print(counts)
    if len(counts) == 0:
        return []
    z = np.polyfit(lengths, counts, 2)
    p = np.poly1d(z)
    
    values = [c - p(l) if l > 1 else 1 for (c,l) in zip(counts,lengths)]
    print(values)
    plot_c_l(counts,lengths)
    best_index = np.argmax(values)
    word = [word for word, count in lcs.most_common()][best_index]
    if lengths[best_index] <= 1:
        return []
    return [word]

def save_catalog(name, catalog):
    with open(f'{name}.csv', 'w') as f:
        writer = csv.writer(f)
        for row in catalog:
            writer.writerow(row.tolist())

def import_catalog(name, agent):
    if not os.path.isfile(f"{name}.csv"):
        print(f"WARNING: The catalog {name} does not exist")
        return
    with open(f'{name}.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            agent.increase_catalog(row)

def plot_c_l(counts, lengths):
    plt.scatter(lengths, counts)
    z = np.polyfit(lengths, counts, 2)
    p = np.poly1d(z)
    plt.plot(range(0,max(lengths)+1),p(range(0,max(lengths)+1)),"r--")
    plt.show()


#[(w,1)]
def graph_count(lcs):
    #word
    counts = [count for (word,count) in lcs.most_common()]
    lengths = [len(word) for (word,count) in lcs.most_common()]
    plot_c_l(counts,lengths)

# string replace, but for lists
def remove_sublist(lst, sub):
    i = 0
    out = []
    while i < len(lst):
        if lst[i:i+len(sub)] == sub:
            i += len(sub)
        else:
            out.append(lst[i])
            i += 1
    return out

# Removes stupid things such as alternating "l", "r" and going right until you get back to the starting position
def cleanse_transition(transition, width):
    return remove_sublist(remove_sublist(remove_sublist(remove_sublist(transition,[2,3]),[3,2]),[3]*width),[2]*width)

def ungroup_transition(grouped_transition, width, catalog):
    transition = []
    env_loc = 0
    for action in grouped_transition:
        loc = action % width
        block = action // width
        while loc != env_loc:
            direction = 1 if abs((loc - env_loc) % width) < abs((env_loc - loc) % width) else -1
            env_loc = (env_loc + direction) % width
            transition.append([None,3,2][direction])
        # I'm sorry
        transition.append(([0,1] + [a + 4 for a in range(len(catalog))])[block])
    return transition

def abstraction_value(evaluation_network, epoch):
    expected_reward = 0

def make_sequence_relative(action_sequence, width):
    init_loc = action_sequence[0] % width
    rel_sequence = []
    for transition in action_sequence:
        if transition == None:
            continue
        action = int(transition)
        loc = (action - init_loc) % width
        block = action // width
        rel_sequence.append(loc + width * block)
    return rel_sequence

def group_transition(ungrouped_transition, width, loc):
    grouped_transition = []
    for a in ungrouped_transition: # a = v,h,l,r
        assert a in range(4), f"Invalid action {a}"
        if a == 0:
            grouped_transition.append((loc % width) + width * 0)
        elif a == 1:
            grouped_transition.append((loc % width) + width * 1)
        elif a == 2:
            loc -= 1
        else:
            loc += 1
    return grouped_transition

# Extracts primitives from catalog actions
def flatten_sequence(action_sequence, width, catalog, std_blocks = 2):
    flattened_sequence = []
    for a in action_sequence:
        if a in range(width*std_blocks):
            flattened_sequence.append(a)
        else:
            c = catalog[a // width - std_blocks]
            l = a % width
            flattened_sequence.extend(group_transition(c, width, l))
    return flattened_sequence

def find_candidates(epochs, width, catalog):
    # Find sequences and make them relative
    solutions = []
    for epoch in epochs:
        sequence = []
        for transition in epoch:
            sequence.append(transition.a)
        solutions.append(make_sequence_relative(sequence, width))
    # Find common sequences
    solutions = [[int(s) for s in ss] for ss in solutions]
    print(solutions)
    candidates = collections.Counter()
    # Used to check if a candidate is already in the catalog
    taken = [tuple(group_transition(c.tolist(),width,0)) for c in catalog]
    n = len(solutions)
    for i in range(n):
        for j in range(n):
            for common_sequence in findCS(solutions[i], solutions[j]):
                key = tuple(flatten_sequence(common_sequence, width, catalog))
                if key in taken or key == ():
                    continue
                candidates[key] += 1
    return candidates

def generate_abstraction(epochs, width, catalog, grouped = True):
    assert grouped, f"non grouped abstractions is not implemented"
    candidates = find_candidates(epochs, width, catalog)
    print(candidates)
    if len(candidates.items()) <= 0:
        return []
    ratings = [len(candidate) * amount for candidate, amount in candidates.items()]
    i = np.argmax(ratings)
    print(ratings)
    if len(list(candidates.keys())[i]) <= 1:
        return []
    return [ungroup_transition(list(candidates.keys())[i], width, catalog)]

#<deprecated>
def get_abstract(epochs, width, catalog, grouped = False):
    sols = []
    for epoch in epochs:
        trans = []
        for transition in epoch:
            actions = transition.a
            print(actions)
            if actions != None:
                trans.append(int(actions))
        if not grouped:
            trans = cleanse_transition(trans, width)
        else:
            trans = ungroup_transition(trans, width, catalog)
        sols.append(trans)

    lcs = collections.Counter()
    print(sols)
    for i in range(len(sols)):
        for j in range(i, len(sols)):
            for x in findCS(sols[i], sols[j]):
                lcs[x] += 1
    words = decide_abstraction(lcs,1)
    # MAKE THIS CORRECT
    #graph_count(lcs)
    return words


def find_bad_abstractions(actions, epsilon, num_std_blocks, num_actions, catalog_size, grouped = True):
    assert grouped, "Ungrouped is not implemented"
    width = num_actions // num_std_blocks
    catalog_actions = [action // width for action in actions if action >= num_actions]
    unique, counts = np.unique(catalog_actions, return_counts=True)
    uses = dict(zip(unique, counts))
    # Bad abstraction if you are used fewer times than random exploration
    count_threshold = int(epsilon * len(actions) * 1/(catalog_size + num_std_blocks))
    print(f'Minimum threshold for if an abstraction is useful {count_threshold}')
    print(f'Current amount of uses {uses}')

    return [act for act in unique if uses[act] < count_threshold and catalog_size > act - num_std_blocks]