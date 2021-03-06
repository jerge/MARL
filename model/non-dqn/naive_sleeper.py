import pandas as pd
import collections
import csv
import matplotlib.pyplot as plt
import numpy as np

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
    # Currently just taking most common words
    good_abstractions = []
    for i in range(2,8):
        ws = [word for word, count in lcs.most_common() if len(word) == i]
        cs = [count for word, count in lcs.most_common() if len(word) == i]
        good_abstractions.append(ws[np.argmax(cs)])
    print(good_abstractions)
    #filtered_dict = {word: count for word, count in lcs.most_common() if len(word) >= min_threshold}
    return good_abstractions#list(filtered_dict.keys())[:n]

def create_catalog(words):
    with open('catalog.csv', 'a') as f:
        writer = csv.writer(f)
        for word in words:
            writer.writerow(word)

#[(w,1)]
def graph_count(lcs):
    #word
    counts = [count for (word,count) in lcs.most_common()]
    lengths = [len(word) for (word,count) in lcs.most_common()]
    plt.scatter(lengths, counts)
    z = np.polyfit(lengths, counts, 2)
    p = np.poly1d(z)
    plt.plot(range(0,10),p(range(0,10)),"r--")
    plt.show()


df = pd.read_csv('bruteforce.csv')

sols = [str(sol).strip().replace(",", "").replace(" ", "").replace("(", "").replace(")", "") for sol in df['actions'] if sol != "()"]
# Maximum string length
N = 100
L = [[0 for i in range(N)]
        for j in range(N)]
lcs = collections.Counter()
for i in range(len(sols)): 
    for j in range(i, len(sols)):
        for x in findLCS(sols[i], sols[j], len(sols[i]), len(sols[j])):
            lcs[x] += 1

words = decide_abstraction(lcs,3)
create_catalog(words)