## 1. basic operators of pv-dict
import pandas as pd
def DPop(x):
    if len(x) == 0:
        return dict()
    i1 = min(x.keys())
    i2 = max(x.keys())
    return {i: x.get(i, 0) for i in range(i1, i2 + 1)}

def DSumMo(x):
    a = 0
    for i, j in x.items():
        a+=i * j
    return a

def DSumV(x):
    return sum(x.values())

def DNegV(d):
    if pd.isnull(d):
        return dict()
    return {i: -j for i, j in d.items()}

def DAdd(d1, d2):
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

def DSub(d1, d2):
    return DAdd(d1, DNegV(d2))

def Dcut(d, l1=None, l2=None):
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

def DFPos(d):
    return {i: j for i, j in d.items() if j > 0}

def DFNeg(d):
    return {i: j for i, j in d.items() if j < 0}

def DFNozero(d):
    return {i: j for i, j in d.items() if j != 0}

def DMin(d1, d2):
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



