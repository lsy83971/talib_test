import pandas as pd
import numpy as np
import random
import math

def s(x):
    if x==0:
        return 0
    return x*math.log(x)

def si_ent(G):
    return sum([s(k) for k in G.ravel()]) / G.sum()

def tri_ent(G):
    ent0 = sum([s(k) for k in G.ravel()])
    ent1 = sum([s(k) for i in range(3) for j in G.sum(axis = i) for k in j])
    ax2 = [(i, j) for i in range(3) for j in range(3) if i < j]
    ent2 = sum([s(j) for i in ax2 for j in G.sum(axis = i)])
    ent3 = s(G.sum(axis = (1, 2, 0)))
    return (ent0 - ent1 + ent2 - ent3) / G.sum(axis = (0, 1, 2))


def db_ent(G):
    ent0 = sum([s(k) for k in G.ravel()])
    ent1 = sum([s(j) for i in range(2) for j in G.sum(axis = i)])
    ent2 = s(G.sum(axis = (1, 0)))
    return (ent0 - ent1 + ent2) / G.sum(axis = (0, 1))



if __name__ == "__main__":
    rate_1d_1 = pd.DataFrame([[1, 1.5, 2, 5]])
    rate_1d_2 = pd.DataFrame([[1, 10.99, 20, 0.8]])

    amt_1d_1 = pd.DataFrame([[1, 1.5, 10, 2.5]])
    amt_1d_2 = pd.DataFrame([[1, 2, 2, 0.8]])

    rate_2d = rate_1d_1.T@rate_1d_2
    amt_2d = amt_1d_1.T@amt_1d_2

    G = np.array([(amt_2d * rate_2d/ (rate_2d + 1)).values, (amt_2d * 1 / (rate_2d + 1)).values])
    G1 = G.copy().ravel()
    random.shuffle(G1)
    G2 = G1.reshape(G.shape)
    tri_ent(G)
    tri_ent(G)

