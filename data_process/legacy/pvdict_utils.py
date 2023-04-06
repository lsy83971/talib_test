import pandas as pd
import numpy as np

## 1. basic operators of pv-dict
def sum_amt(x):
    a = 0
    for i, j in x.items():
        a+=i * j
    return a

def sum_volume(x):
    return sum(x.values())

def inverse_dict(d):
    if pd.isnull(d):
        return dict()
    return {i: -j for i, j in d.items()}

def add_dict(d1, d2):
    if pd.isnull(d1):
        d1 = dict()
    if pd.isnull(d2):
        d2 = dict()
    d3 = d1.copy()
    for i, j in d2.items():
        if i in d1:
            d3[i] += j
        else:
            d3[i] = j
    return d3

def sub_dict(d1, d2):
    return add_dict(d1, inverse_dict(d2))

def cut_dict(d, l1=None, l2=None):
    if pd.isnull(d):
        return dict()
    d1 = d.copy()
    if l1 is not None:
        for i in d:
            if i < l1:
                del d1[i]
    if l2 is not None:
        for i in d:
            if i > l2:
                del d1[i]
    return d1

def pos_dict(d):
    return {i: j for i, j in d.items() if j > 0}

def neg_dict(d):
    return {i: j for i, j in d.items() if j < 0}

def nozero_dict(d):
    return {i: j for i, j in d.items() if j != 0}

def min_dict(d1, d2):
    if pd.isnull(d1):
        d1 = dict()
    if pd.isnull(d2):
        d2 = dict()
    d3 = dict()
    for i, j in d2.items():
        if i in d1:
            d3[i] = min(j, d1[i])
        else:
            d3[i] = 0
    return d3


