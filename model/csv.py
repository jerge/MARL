import csv 
import pandas as pd
with open('bruteforce.csv,w) as f:
           writer = csv.writer(f)



def writeTo(name, actions):
    row = [name, actions]
    writeer.writerow(row)


def getSolution(name):
    df = pd.read_csv(f)
    index = df.loc[df['0'] == name]
    return df.at['index', 1]
