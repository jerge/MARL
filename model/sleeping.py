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

    print([word for word, count in lcs.most_common() if len(word) > 1][:n])
    return [word for word, count in lcs.most_common() if len(word) > 1][:n]
    # for i in range(2,20):
    #     ws = [word for word, count in lcs.most_common() if len(word) == i]
    #     cs = [count for word, count in lcs.most_common() if len(word) == i]
    #     if len(ws) > 0:
    #         good_abstractions.append(ws[np.argmax(cs)])
    # print(good_abstractions)
    # #filtered_dict = {word: count for word, count in lcs.most_common() if len(word) >= min_threshold}
    # # NOTE: CAN BE NONE
    # return good_abstractions#list(filtered_dict.keys())[:n]

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

#[(w,1)]
def graph_count(lcs):
    #word
    counts = [count for (word,count) in lcs.most_common()]
    lengths = [len(word) for (word,count) in lcs.most_common()]
    plt.scatter(lengths, counts)
    z = np.polyfit(lengths, counts, 2)
    p = np.poly1d(z)
    plt.plot(range(0,max(lengths)),p(range(0,max(lengths))),"r--")
    plt.show()

#globtrans, =([s,m,a,r,t])

# Removes stupid things such as alternating "l", "r" and going right until you get back to the starting position
def cleanse_transition(transition, width):
    return transition.replace("23","").replace("32","").replace("3"*width, "").replace("2"*width,"")

def get_abstract(epochs, width):
    sols = []
    for epoch in epochs:
        trans = ""
        for transition in epoch:
            actions = transition.a
            if actions != None:
                trans = trans + (str(int(actions)))
        sols.append(cleanse_transition(trans,width))

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
                lcs[x] += 1
    words = decide_abstraction(lcs,1)
    #graph_count(lcs)
    return words
