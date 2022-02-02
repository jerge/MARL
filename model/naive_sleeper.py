import pandas as pd

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
import collections
df = pd.read_csv('bruteforce.csv')

sols = [str(sol).strip().replace(",", "").replace(" ", "").replace("(", "").replace(")", "") for sol in df['actions'] if sol != "()"]
# Maximum string length
N = 90
L = [[0 for i in range(N)]
        for j in range(N)]
lcs = collections.Counter()
for i in range(len(sols)): 
    for j in range(i, len(sols)):
        for x in findLCS(sols[i], sols[j], len(sols[i]), len(sols[j])):
            lcs[x] += 1
print(lcs)
