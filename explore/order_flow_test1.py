import matplotlib.pyplot as plt
import pickle
import math
import pandas as pd
pd.set_option("display.max_rows",500)
import numpy as np
import math
import sys
import os

os.chdir("/home/lishiyu/Project")
sys.path.append("./bin_tools")
from importlib import reload
import bin_tools.bins
reload(bin_tools.bins)
from bin_tools.bins import binning, bins_simple_mean
import warnings
warnings.filterwarnings("ignore")
rb_fdata=pickle.load(open("./explore/rb_raw_feature.pkl","rb"))




df=pd.concat(list(rb_fdata[('2022-12-19', 'rb2305')]. values())
             + list(rb_fdata[('2022-12-20', 'rb2305')]. values())
             + list(rb_fdata[('2022-12-21', 'rb2305')]. values())             
             )
df = df.reset_index(drop=True)

df["Ask1_up"]=df[["AskPrice1_Lag1","AskPrice1"]].apply(max,axis=1)
df["Bid1_low"]=df[["BidPrice1_Lag1","BidPrice1"]].apply(min,axis=1)

df["Turnover_DiffLen1S"]=df["Turnover_DiffLen1"]/10
df["Tsp_Diff1"]=df["ExchTimeOffsetUs"]-df["ExchTimeOffsetUs"].shift(1)-0.5
df["BidPrice_Intv"]=(df["BidPrice1"]-df["BidPrice5"]-4)
df["AskPrice_Intv"]=(df["AskPrice5"]-df["AskPrice1"]-4)
df["AskBidPrice_Intv"]=(df["AskPrice1"]-df["BidPrice1"]-1)

df["AskBidPrice_Intv_Lag1"]=df["AskBidPrice_Intv"].shift(1)
df["AskPrice_Intv_Lag1"]=df["AskPrice_Intv"].shift(1)
df["BidPrice_Intv_Lag1"]=df["BidPrice_Intv"].shift(1)
df["BidPrice1_Diff1"]=df["BidPrice1"]-df["BidPrice1"].shift(1)
df["MidPriceLvl1_Lag1"] = df["MidPriceLvl1"]. shift(1)
df["Price1_span"] = (df["Ask1_up"] - df["Bid1_low"])
df["Price1_mid"] = (df["Ask1_up"] + df["Bid1_low"]) / 2
df["Volume_DiffLen1_Lag1"] = df["Volume_DiffLen1"]. shift(1)

df["AskBid1_minv_Lag1"] = df.apply(lambda x:min(x["AskVolume1_Lag1"], x["BidVolume1_Lag1"]), axis=1)
df["AskBid1_maxv_Lag1"] = df.apply(lambda x:max(x["AskVolume1_Lag1"], x["BidVolume1_Lag1"]), axis=1)

df["ReturnLen1"] = df["ReturnLen1"]. fillna(0)
df["ReturnLen20"] = df["ReturnLen20"]. fillna(0)
df["ReturnLen40"] = df["ReturnLen40"]. fillna(0)
df["ReturnLen60"] = df["ReturnLen60"]. fillna(0)

def guess_exchange1(x):
    v = x["Volume_DiffLen1"]
    a = x["Turnover_DiffLen1S"]
    if a == 0:
        return dict()
    if pd.isnull(v):
        return dict()
    p1 = math.floor(a / v)
    if p1 > x["Ask1_up"]:
        return {x["Ask1_up"]:  v}
    if p1 < x["Bid1_low"]:
        return {x["Bid1_low"]:  v}
    p2 = p1 + 1
    v2 = (a - p1 * v)
    v1 = v - v2
    return {p1: v1, p2: v2}

def sum_amt(x):
    a = 0
    for i, j in x.items():
        a+=i * j
    return a
df["ExchTab"] = df.apply(guess_exchange1, axis=1)
df["AmtDev"] = df["ExchTab"]. apply(sum_amt) - df["Turnover_DiffLen1S"]

from collections import defaultdict
def ask_dict(x):
    x1 = defaultdict(int)
    x1[x["AskPrice1"]] = x["AskVolume1"]
    x1[x["AskPrice2"]] = x["AskVolume2"]
    x1[x["AskPrice3"]] = x["AskVolume3"]
    x1[x["AskPrice4"]] = x["AskVolume4"]
    x1[x["AskPrice5"]] = x["AskVolume5"]
    return x1
def bid_dict(x):
    x1 = defaultdict(int)
    x1[x["BidPrice1"]] = x["BidVolume1"]
    x1[x["BidPrice2"]] = x["BidVolume2"]
    x1[x["BidPrice3"]] = x["BidVolume3"]
    x1[x["BidPrice4"]] = x["BidVolume4"]
    x1[x["BidPrice5"]] = x["BidVolume5"]
    return x1
def ask_dict_lag1(x):
    x1 = defaultdict(int)
    x1[x["AskPrice1_Lag1"]] = x["AskVolume1_Lag1"]
    x1[x["AskPrice2_Lag1"]] = x["AskVolume2_Lag1"]
    x1[x["AskPrice3_Lag1"]] = x["AskVolume3_Lag1"]
    x1[x["AskPrice4_Lag1"]] = x["AskVolume4_Lag1"]
    x1[x["AskPrice5_Lag1"]] = x["AskVolume5_Lag1"]
    return x1
def bid_dict_lag1(x):
    x1 = defaultdict(int)
    x1[x["BidPrice1_Lag1"]] = x["BidVolume1_Lag1"]
    x1[x["BidPrice2_Lag1"]] = x["BidVolume2_Lag1"]
    x1[x["BidPrice3_Lag1"]] = x["BidVolume3_Lag1"]
    x1[x["BidPrice4_Lag1"]] = x["BidVolume4_Lag1"]
    x1[x["BidPrice5_Lag1"]] = x["BidVolume5_Lag1"]
    return x1

def inverse_dict(d):
    return {i: -j for i, j in d.items()}

def merge_dict(d1, d2):
    d3 = d1.copy()
    for i, j in d2.items():
        if i in d1:
            d3[i] += j
        else:
            d3[i] = j
    return d3
            
def cut_dict(d, l1=None, l2=None):
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

def calc_exch_tab(x):
    ask_dict = {
        x["AskPrice1"]: x["AskVolume1"], 
        x["AskPrice2"]: x["AskVolume2"], 
        x["AskPrice3"]: x["AskVolume3"], 
        x["AskPrice4"]: x["AskVolume4"], 
        x["AskPrice5"]: x["AskVolume5"], 
                }
    ask_dict_lag1 = {
        x["AskPrice1_Lag1"]: x["AskVolume1_Lag1"], 
        x["AskPrice2_Lag1"]: x["AskVolume2_Lag1"], 
        x["AskPrice3_Lag1"]: x["AskVolume3_Lag1"], 
        x["AskPrice4_Lag1"]: x["AskVolume4_Lag1"], 
        x["AskPrice5_Lag1"]: x["AskVolume5_Lag1"], 
                }
    bid_dict = {
        x["BidPrice1"]: x["BidVolume1"], 
        x["BidPrice2"]: x["BidVolume2"], 
        x["BidPrice3"]: x["BidVolume3"], 
        x["BidPrice4"]: x["BidVolume4"], 
        x["BidPrice5"]: x["BidVolume5"], 
                }
    bid_dict_lag1 = {
        x["BidPrice1_Lag1"]: x["BidVolume1_Lag1"], 
        x["BidPrice2_Lag1"]: x["BidVolume2_Lag1"], 
        x["BidPrice3_Lag1"]: x["BidVolume3_Lag1"], 
        x["BidPrice4_Lag1"]: x["BidVolume4_Lag1"], 
        x["BidPrice5_Lag1"]: x["BidVolume5_Lag1"], 
                }

    ask1_up = max(x["AskPrice1_Lag1"], x["AskPrice1_Lag1"])
    ask1_low = min(x["AskPrice1_Lag1"], x["AskPrice1_Lag1"])
    bid1_up = max(x["BidPrice1_Lag1"], x["BidPrice1_Lag1"])
    bid1_low = min(x["BidPrice1_Lag1"], x["BidPrice1_Lag1"])
    
    ask5_up = max(x["AskPrice5_Lag1"], x["AskPrice5_Lag1"])
    ask5_low = min(x["AskPrice5_Lag1"], x["AskPrice5_Lag1"])
    bid5_up = max(x["BidPrice5_Lag1"], x["BidPrice5_Lag1"])
    bid5_low = min(x["BidPrice5_Lag1"], x["BidPrice5_Lag1"])

    
    v = x["Volume_DiffLen1"]
    a = x["Turnover_DiffLen1S"]
    if a == 0:
        ex_dict = dict()
    elif pd.isnull(v):
        ex_dict = dict()
    else:
        p1 = math.floor(a / v)
        if p1 > x["Ask1_up"]:
            ex_dict = {x["Ask1_up"]:  v}
        elif p1 < x["Bid1_low"]:
            ex_dict = {x["Bid1_low"]:  v}
        else:
            p2 = p1 + 1
            v2 = (a - p1 * v)
            v1 = v - v2
            ex_dict = {p1: v1, p2: v2}

    ask_dict_addex = merge_dict(ask_dict, ex_dict)
    ask_dict_addex_cut = cut_dict(ask_dict_addex, l2=ask5_low)
    ask_dict_lag1_cut = cut_dict(ask_dict_lag1, l2=ask5_low)
    ask_dict_dif = merge_dict(ask_dict_addex_cut, inverse_dict(ask_dict_lag1_cut))

    bid_dict_addex = merge_dict(bid_dict, ex_dict)
    bid_dict_addex_cut = cut_dict(bid_dict_addex, l1=bid5_up)
    bid_dict_lag1_cut = cut_dict(bid_dict_lag1, l1=bid5_up)
    bid_dict_dif = merge_dict(bid_dict_addex_cut, inverse_dict(bid_dict_lag1_cut))

    return {
        "ask1_up": ask1_up, 
        "ask1_low": ask1_low, 
        "ask5_up": ask5_up, 
        "ask5_low": ask5_low,
        "ask1_up": ask1_up, 
        "ask1_low": ask1_low, 
        "ask1_up": ask1_up, 
        "ask1_low": ask1_low, 
        
        "ex_dict": ex_dict, 
        
        "ask_dict": ask_dict,
        "ask_dict_lag1": ask_dict_lag1,
        "ask_dict_lag1_cut": ask_dict_lag1_cut,
        "ask_dict_addex": ask_dict_addex,
        "ask_dict_addex_cut": ask_dict_addex_cut,
        "ask_dict_dif": ask_dict_dif,

        "bid_dict": bid_dict,
        "bid_dict_lag1": bid_dict_lag1,
        "bid_dict_lag1_cut": bid_dict_lag1_cut,
        "bid_dict_addex": bid_dict_addex,
        "bid_dict_addex_cut": bid_dict_addex_cut,
        "bid_dict_dif": bid_dict_dif,                                
           }







df["AskDict"] = df.apply(ask_dict, axis=1)
df["BidDict"] = df.apply(bid_dict, axis=1)
df["AskDict_Lag1"] = df.apply(ask_dict_lag1, axis=1)
df["BidDict_Lag1"] = df.apply(bid_dict_lag1, axis=1)

df["Ask5_up"] = df.apply(lambda x:max(x["AskPrice5_Lag1"], x["AskPrice5"]), axis=1)
df["Bid5_low"] = df.apply(lambda x:min(x["BidPrice5_Lag1"], x["BidPrice5"]), axis=1)

df["res"] = df.apply(calc_exch_tab, axis=1)


_df = pd.DataFrame(list(df['res']))
df = pd.concat([df, _df], axis=1)

#  A1 -> A1
#  B1 -> B1
cond1 = (df["AskBidPrice_Intv_Lag1"]==0)&\
  (df["AskBidPrice_Intv"]==0)&\
  (df["BidPrice1_Diff1"]==0)
#     -> A1
#  A1 -> B1
#  B1 
cond2 = (df["AskBidPrice_Intv_Lag1"]==0)&\
  (df["AskBidPrice_Intv"]==0)&\
  (df["BidPrice1_Diff1"]==1)
#  A1
#  B1 -> A1
#     -> B1
cond3 = (df["AskBidPrice_Intv_Lag1"]==0)&\
  (df["AskBidPrice_Intv"]==0)&\
  (df["BidPrice1_Diff1"]== -1)
#     -> A1
#  A1 
#  B1 -> B1

cond4 = (df["AskBidPrice_Intv_Lag1"]==0)&\
  (df["AskBidPrice_Intv"]==1)&\
  (df["BidPrice1_Diff1"]==0)
#  A1 -> A1
#  B1 
#     -> B1
cond5 = (df["AskBidPrice_Intv_Lag1"]==0)&\
  (df["AskBidPrice_Intv"]==1)&\
  (df["BidPrice1_Diff1"]== -1)
#  A1 
#     -> A1
#  B1 -> B1
cond6 = (df["AskBidPrice_Intv_Lag1"]==1)&\
  (df["AskBidPrice_Intv"]==0)&\
  (df["BidPrice1_Diff1"]==0)
#  A1 -> A1
#     -> B1
#  B1 
cond7 = (df["AskBidPrice_Intv_Lag1"]==1)&\
  (df["AskBidPrice_Intv"]==0)&\
  (df["BidPrice1_Diff1"]== 1)

df["TabType"] = -1
df.loc[cond1, "TabType"] = 1
df.loc[cond2, "TabType"] = 2
df.loc[cond3, "TabType"] = 3
df.loc[cond4, "TabType"] = 4
df.loc[cond5, "TabType"] = 5
df.loc[cond6, "TabType"] = 6
df.loc[cond7, "TabType"] = 7

df1 = df[cond1]
df2 = df[cond2]
df3 = df[cond3]
df4 = df[cond4]
df5 = df[cond5]
df6 = df[cond6]
df7 = df[cond7]



##################
from bin_tools.bins import *
df1["ReturnLen40"]. value_counts().sort_index()


def cond1_var(x):
    p1 = x["BidPrice1"]
    p2 = p1 + 1

    ask_dict_dif_pos = pos_dict(x["ask_dict_dif"])
    ask_dict_dif_neg = neg_dict(x["ask_dict_dif"])

    ask2_addord = sum(cut_dict(x["ask_dict_dif"], l2=p2 + 2).values())
    ask1_addord = sum(cut_dict(x["ask_dict_dif"], l2=p2 + 1).values())
    ask_addord = sum(cut_dict(x["ask_dict_dif"], l2=p2).values())

    ask2_addord_pos = sum(cut_dict(ask_dict_dif_pos, l2=p2 + 2).values())
    ask1_addord_pos = sum(cut_dict(ask_dict_dif_pos, l2=p2 + 1).values())
    ask_addord_pos = sum(cut_dict(ask_dict_dif_pos, l2=p2).values())

    ask2_addord_neg = sum(cut_dict(ask_dict_dif_neg, l2=p2 + 2).values())
    ask1_addord_neg = sum(cut_dict(ask_dict_dif_neg, l2=p2 + 1).values())
    ask_addord_neg = sum(cut_dict(ask_dict_dif_neg, l2=p2).values())

    bid_dict_dif_pos = pos_dict(x["bid_dict_dif"])
    bid_dict_dif_neg = neg_dict(x["bid_dict_dif"])

    bid2_addord = sum(cut_dict(x["bid_dict_dif"], l1=p1 - 2).values())
    bid1_addord = sum(cut_dict(x["bid_dict_dif"], l1=p1 - 1).values())
    bid_addord = sum(cut_dict(x["bid_dict_dif"], l1=p1).values())

    bid2_addord_pos = sum(cut_dict(bid_dict_dif_pos, l1=p1 - 2).values())
    bid1_addord_pos = sum(cut_dict(bid_dict_dif_pos, l1=p1 - 1).values())
    bid_addord_pos = sum(cut_dict(bid_dict_dif_pos, l1=p1).values())

    bid2_addord_neg = sum(cut_dict(bid_dict_dif_neg, l1=p1 - 2).values())
    bid1_addord_neg = sum(cut_dict(bid_dict_dif_neg, l1=p1 - 1).values())
    bid_addord_neg = sum(cut_dict(bid_dict_dif_neg, l1=p1).values())

    return {
        "ask_dict_dif_pos": ask_dict_dif_pos, 
        "ask_dict_dif_neg": ask_dict_dif_neg, 
        "ask2_addord": ask2_addord,
        "ask1_addord": ask1_addord,
        "ask_addord": ask_addord,    
        "ask2_addord_pos": ask2_addord_pos,
        "ask1_addord_pos": ask1_addord_pos,
        "ask_addord_pos": ask_addord_pos,    
        "ask2_addord_neg": ask2_addord_neg,
        "ask1_addord_neg": ask1_addord_neg,
        "ask_addord_neg": ask_addord_neg,    
        "bid_dict_dif_pos": bid_dict_dif_pos, 
        "bid_dict_dif_neg": bid_dict_dif_neg, 
        "bid2_addord": bid2_addord,
        "bid1_addord": bid1_addord,
        "bid_addord": bid_addord,    
        "bid2_addord_pos": bid2_addord_pos,
        "bid1_addord_pos": bid1_addord_pos,
        "bid_addord_pos": bid_addord_pos,    
        "bid2_addord_neg": bid2_addord_neg,
        "bid1_addord_neg": bid1_addord_neg,
        "bid_addord_neg": bid_addord_neg,    

            }


df1_var = pd.DataFrame(list(df1.apply(cond1_var, axis=1)))
df1_var.index = df1.index
df1 = pd.concat([df1, df1_var], axis=1)



#########################################################
# df1.b2(x="bid_addord", y="ReturnLen40", f=f_med)      #
# df1.b2(x="ask2_addord", y="ReturnLen40", f=f_med)     #
# df1.b2(x="ask1_addord", y="ReturnLen40", f=f_med)     #
# df1.b2(x="ask_addord", y="ReturnLen40", f=f_med)      #
#                                                       #
#                                                       #
# df1.b2(x="ask_addord_pos", y="ReturnLen40", f=f_med)  #
# df1.b2(x="ask_addord_neg", y="ReturnLen40", f=f_med)  #
#                                                       #
# df1["test"] = (df1["ask_addord"] - df1["bid_addord"]) #
# df1.b2(x="test", y="ReturnLen20", f=f_med, quant=20)  #
# df1.b2(x="test", y="ReturnLen60", f=f_med, quant=20)  #
#########################################################











