import pandas as pd
import collections
import csv
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import os

def get_solution(name):
    df = pd.read_csv(f)
    index = df.loc[df['0'] == name]
    return df.at['index', 1]

# Returns all common sequences in x and y of the largest length
def findCS(x,y):
    parts = [x[a:b] for a, b in combinations(range(len(x) + 1), r = 2)]
    common = set([part for part in parts if part in y])
    if len(common) == 0:
        return []
    max_len = max([len(c) for c in common])
    return [c for c in common if len(c) == max_len]

def decide_abstraction(lcs,n):
    # Currently just taking n most common words
    print(lcs)
    lengths = [word.count('0') + word.count('1') for (word,count) in lcs.most_common()]
    counts = [count for (word,count) in lcs.most_common()]
    print(lengths)
    print(counts)
    z = np.polyfit(lengths, counts, 2)
    p = np.poly1d(z)
    
    values = [c - p(l) if l > 1 else 1 for (c,l) in zip(counts,lengths)]
    print(values)
    #plot_c_l(counts,lengths)
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

#globtrans, =([s,m,a,r,t])

# Removes stupid things such as alternating "l", "r" and going right until you get back to the starting position
def cleanse_transition(transition, width):
    return transition.replace("23","").replace("32","").replace("3"*width, "").replace("2"*width,"")

def ungroup_transition(grouped_transition, width):
    transition = ""
    env_loc = 0
    for action in grouped_transition:
        loc = int(action) % width
        block = int(action) // width
        while loc != env_loc:
            direction = 1 if abs((loc - env_loc) % width) < abs((env_loc - loc) % width) else -1
            env_loc = (env_loc + direction) % width
            transition = transition + " 32"[direction]
        transition = transition + "01"[block]
    # Remove leading or ending sidestepping
    transition = transition.strip('23')
    return transition


def get_abstract(epochs, width, grouped = False):
    sols = ['1220', '1220', '1220', '0', '0', '1331', '0', '020', '020', '0', '1220', '1', '1220', '020', '1220', '1331', '0', '1', '0', '1221', '030', '0', '1', '1', '020', '1220', '1220', '1221', '1220', '1331', '1', '020', '1221', '1', '0', '020', '0', '030', '1', '1', '030', '1220', '1', '1', '1', '1', '1221', '1220', '030', '1220', '1', '020', '0', '1331', '1', '0', '1', '1221', '030', '020', '1', '1220', '030', '030', '1', '1220', '1221', '1221', '0', '030', '1331', '1', '0', '030', '030', '1', '0', '1220', '0', '0', '11', '1221', '0', '1221', '0', '1', '1331', '030', '0', '1331', '0', '0', '0', '1221', '020', '0', '020', '0', '0', '1', '0', '1221', '1220', '0', '020', '1', '1', '0', '1221', '0', '0', '0', '020', '0', '1', '1', '0', '030', '0', '0', '1', '030', '020', '0', '0', '1331', '020', '1', '020', '1221', '0', '1331', '0', '1221', '1220', '0', '1', '0', '0', '030', '0', '0', '1', '020', '020', '020', '1', '030', '1221', '0']
    # Maximum string length
    N = 100
    global L
    L = [[0 for i in range(N)]
            for j in range(N)]
    lcs = collections.Counter()
    print(sols)
    for i in range(len(sols)):
        for j in range(i, len(sols)):
            for x in findCS(sols[i], sols[j]):
                lcs[x] += 1
    words = decide_abstraction(lcs,1)
    return words

def find_bad_abstractions(actions, epsilon, num_std_blocks, num_actions, catalog_size, grouped = True):
    assert grouped, "Ungrouped is not implemented"
    width = num_actions // num_std_blocks
    catalog_actions = [action // width for action in actions if action >= num_actions]
    unique, counts = np.unique(catalog_actions, return_counts=True)
    uses = dict(zip(unique, counts))
    # Bad abstraction if you are used fewer times than random exploration
    count_threshold = int(epsilon * len(actions) * 1/(catalog_size + num_std_blocks))
    print(count_threshold)
    print(uses)

    return [act for act in unique if uses[act] < count_threshold and catalog_size > act - num_std_blocks]