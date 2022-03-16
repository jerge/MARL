import pandas as pd
import collections
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def get_solution(name):
    df = pd.read_csv(f)
    index = df.loc[df['0'] == name]
    return df.at['index', 1]



# Returns set containing all LCS
# for X[0..m-1], Y[0..n-1]
def findLCS(x, y, m, n): 
    # construct a set to store possible LCS
    s = set()
 
    # If we reaches end of either string, return
    # a empty set
    if m == 0 or n == 0:
        s.add("")
        return s
 
    # If the last characters of X and Y are same
    if x[m - 1] == y[n - 1]:
 
        # recurse for X[0..m-2] and Y[0..n-2] in
        # the matrix
        tmp = findLCS(x, y, m - 1, n - 1)
 
        # append current character to all possible LCS
        # of substring X[0..m-2] and Y[0..n-2].
        for string in tmp:
            s.add(string + x[m - 1])
 
    # If the last characters of X and Y are not same
    else:
 
        # If LCS can be constructed from top side of
        # the matrix, recurse for X[0..m-2] and Y[0..n-1]
        if L[m - 1][n] >= L[m][n - 1]:
            s = findLCS(x, y, m - 1, n)
 
        # If LCS can be constructed from left side of
        # the matrix, recurse for X[0..m-1] and Y[0..n-2]
        if L[m][n - 1] >= L[m - 1][n]:
            tmp = findLCS(x, y, m, n - 1)
 
            # merge two sets if L[m-1][n] == L[m][n-1]
            # Note s will be empty if L[m-1][n] != L[m][n-1]
            for i in tmp:
                s.add(i)
    return s

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
    sols = []
    for epoch in epochs:
        trans = ""
        for transition in epoch:
            actions = transition.a
            if actions != None:
                trans = trans + (str(int(actions)))
        if not grouped:
            trans = cleanse_transition(trans,width)
        else:
            trans = ungroup_transition(trans,width)
        sols.append(trans)

    # Maximum string length
    N = 100
    global L
    L = [[0 for i in range(N)]
            for j in range(N)]
    lcs = collections.Counter()
    print(sols)
    for i in range(len(sols)):
        for j in range(i, len(sols)):
            for x in findLCS(sols[i], sols[j], len(sols[i]), len(sols[j])):
                if not x == "":
                    lcs[x] += 1
    words = decide_abstraction(lcs,1)
    # MAKE THIS CORRECT
    #graph_count(lcs)
    return words
#h3 h2 h1
#[v0,v1,v2,h1,h2,h3]
#transition = "012020102012121210201020102"
#print(ungroup_transition(transition, 3))