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

def show_price_tab(x, Lag=""):
    idx = ["Ask5", "Ask4", "Ask3", "Ask2", "Ask1",
                             "Bid1", "Bid2", "Bid3", "Bid4", "Bid5",
                             ]
    x1 = pd.DataFrame(index=idx,
                      columns=["Price", "Volume"]
                      )
    for i in x1.index:
        for j in x1.columns:
            x1.loc[i, j] = x.loc[i[:3] + j + i[3:] + Lag]

    return x1




def check_valid(df):
    if df.shape[0]==0:
        return None
    res=dict()

    # 1.检验tick
    数据时间连续性
    tick_time_diff=df["ExchTimeOffsetUs"].diff()
    tick_time_diff.value_counts()
    cond_tick_time_abnormal=(tick_time_diff!=0.5)

    res["tick_abnormal_cnt"]=cond_tick_time_abnormal.sum()
    res["tick_abnormal_time"]=pd.Series(df[cond_tick_time_abnormal].index[1:])

    ## 正常情况相邻orderbook相差0.5s 
    ## 存在极少数(6/8993)相差1s的情况
    ## 相差一秒的样本 暂时未发现其具有特殊共性
    ## 猜测 1.orderbook未发生变动 则不产生新记录？ 2.数据遗漏

    # 2.查看交易窗口变动是否频繁
    price_change=(df["AskPrice1"]-df["AskPrice1_Lag1"])
    res["price_change_cnt"]=price_change.value_counts().sort_index()
    res["price_change_rate"]=(price_change!=0).mean()
    # -5.0       1
    # -4.0       1
    # -3.0       2
    # -2.0      13
    # -1.0     538
    #  0.0    7860
    #  1.0     566
    #  2.0      11

    ## 12.6% 的tick下出现交易窗口变动

    # 3.查看交易窗口价格连续性
    # 同类型求差值
    # ask
    for i in ["Ask","Bid"]:
        for j in range(1,5):
            _dif=(df[f"{i}Price{j+1}"]-df[f"{i}Price{j+1}"])
            res[f"{i}{j}_{j+1}dif_cnt"]=_dif.value_counts().sort_index()
            res[f"{i}{j}_{j+1}dif_rate"]=(_dif>0).mean()

    # 盘口ask1 bid1差值
    _dif=(df["AskPrice1"]-df["BidPrice1"])
    res["AskBid_dif_cnt"]=_dif.value_counts()
    res["AskBid_dif_rate"]=(_dif!=1).mean()
    # 1    8864
    # 2     128
    # 4       1
    res["begin"]=df.iloc[0]
    res["end"]=df.iloc[-1]
    return res

def check_valid_f(fdata,f):
    try:
        res=dict()
        for (i1,i2),j in fdata.items():
            res1=dict()
            res[i1]=res1
            for (k1,k2) in j.items():
                res1[k1]=f(k2)
    except:
        print(i1,k1)
        raise
    return res

def apply_fdata(fdata,f):
    try:
        res=pd.DataFrame()
        for i1,j in fdata.items():
            for (k1,k2) in j.items():
                if k2 is None:
                    res.loc[i1,k1]=None
                else:
                    res.loc[i1,k1]=f(k2)
    except:
        print(i1,k1)
        raise
    return res





cvf=check_valid_f(rb_fdata,check_valid)

apply_fdata(cvf,(lambda x:x["tick_abnormal_cnt"]))
apply_fdata(cvf,(lambda x:round((x["end"]["Turnover"]-x["begin"]["Turnover"])/1.0e+8)))
apply_fdata(cvf,(lambda x:round((x["end"]["OpenInterest"]-x["begin"]["OpenInterest"])/1.0e+2)))

apply_fdata(cvf,(lambda x:round(x["price_change_rate"],2)))
apply_fdata(cvf,(lambda x:round(x["AskBid_dif_rate"],3)))



all_data=pd.concat([j for i in rb_fdata.values() for j in i.values()])
t_data=all_data[["AskPrice1_Lag1","AskPrice1","BidPrice1_Lag1","BidPrice1"]]
t_data=t_data[~t_data["AskPrice1_Lag1"].isnull()]

(t_data["AskPrice1"]-t_data["BidPrice1"]).value_counts()

t_data["dif_Ask"]=(t_data["AskPrice1"]-t_data["AskPrice1_Lag1"])
t_data["dif_Bid"]=(t_data["BidPrice1"]-t_data["BidPrice1_Lag1"])
t_data["dif_AB"]=(t_data["AskPrice1_Lag1"]-t_data["BidPrice1_Lag1"])

t_data.groupby(["dif_AB","dif_Ask","dif_Bid"]).apply(lambda x:x.shape[0])


all_data.columns

(all_data["BidPrice5"]-all_data["BidPrice1"]).value_counts()
(all_data["AskPrice5"]-all_data["AskPrice1"]).value_counts()


(all_data["BidPrice1_Lag1"]==all_data["BidPrice1"])&(all_data["BidPrice1_Lag1"]==all_data["BidPrice1"])



############################################################


df=rb_fdata[('2022-12-19', 'rb2305')]["day2"]

# TradingDay                 2022-12-19
# Symbol                         rb2305
# ExchTimeOffsetUs                32901
# Volume                        1334446
# Turnover                  53036817110
# OpenInterest                  1866256

# BidPrice1                        3962
# BidVolume1                          2




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

(df["Ask1_up"] - df["Bid1_low"]).value_counts().sort_index()

# Turnover_DiffLen1                 NaN
# Volume_DiffLen1                   NaN



## 1.普通情况
df["FLAG_AskBid1_Stable"]=(df["AskBidPrice_Intv_Lag1"]==0)&(df["AskBidPrice_Intv"]==0)&(df["BidPrice1_Diff1"]==0)

df1=df[(df["FLAG_AskBid1_Stable"]==1)]
df1["F_C1_VWAP_Lt_LOW"]=(df1["VWAP"]<df1["BidPrice1"])
df1["F_C1_VWAP_Lt2_LOW"]=(df1["VWAP"]<(df1["BidPrice1"]-1))

df1["F_C1_VWAP_Gt_UP"]=(df1["VWAP"]>df1["AskPrice1"])
df1["F_C1_VWAP_Gt2_UP"]=(df1["VWAP"]>(df1["AskPrice1"]+1))

df1["_v1"]=-(df1["Turnover_DiffLen1S"]-df1["Volume_DiffLen1"]*df1["BidPrice1"])
df1["_v2"]=(df1["Turnover_DiffLen1S"]-df1["Volume_DiffLen1"]*df1["AskPrice1"])

def exchange_volume_guess1(x):
    v0=x["Volume_DiffLen1"]
    if x["Volume_DiffLen1"]==0:
        return []
    if x["F_C1_VWAP_Lt_LOW"] is True:
        p=x["BidPrice1"]
        v1=x["_v1"]
        v1=min(v1,v0)
        return [(p-1,v1),(p,v0-v1)]
    if x["F_C1_VWAP_Gt_UP"] is True:
        p=x["AskPrice1"]
        v2=x["_v2"]
        v2=min(v2,v0)
        return [(p,v0-v2),(p+1,v2)]
    
    p=x["BidPrice1"]
    v2=-x["_v1"]
    v2=min(v2,v0)
    return [(p,v0-v2),(p+1,v2)]

def sum_amt(x):
    a=0
    for i in x:
        a+=i[0]*i[1]
    return a

df1["ExPriceTab"]=df1.apply(exchange_volume_guess1,axis=1)
(df1["ExPriceTab"]. apply(sum_amt) - df1["Turnover_DiffLen1S"]).abs().sum()


## 2.特殊情况
# 
#        *
# * ---> *  or  ---->
# *                  *
#                    *






_df=df[~(df["FLAG_AskBid1_Stable"]==1)]

cond2=(_df["AskBidPrice_Intv_Lag1"]==0)&(_df["AskBidPrice_Intv"]==0)&((_df["BidPrice1_Diff1"]==1)|(_df["BidPrice1_Diff1"]==-1))

df2=_df[cond2]
df2["F_C2"]=True
df2["F_C2_UP"] = (df2["BidPrice1_Diff1"]==1)
df2["F_C2_VWAP_Gt_UP"]=(df2["VWAP"]>df2["Ask1_up"])
df2["F_C2_VWAP_Lt_LOW"]=(df2["VWAP"]<df2["Bid1_low"])


df2_up = df2.loc[df2["F_C2_UP"]]
df2_down = df2.loc[~df2["F_C2_UP"]]

(df2_up["MidPriceLvl1_Lag1"] - df2_up["VWAP"]).abs().mean()
(df2_up["MidPriceLvl1"] - df2_up["VWAP"]).abs().mean()


(df2_down["MidPriceLvl1_Lag1"] - df2_down["VWAP"]).abs().mean()
(df2_down["MidPriceLvl1"] - df2_down["VWAP"]).abs().mean()

df2.columns

def exchange_volume_guess2(x):
    v0=x["Volume_DiffLen1"]
    if x["Volume_DiffLen1"]==0:
        return []
    if x["F_C2_VWAP_Gt_UP"] is True:
        p=x["Ask1_up"]
        return [(p,v0)]
    if x["F_C2_VWAP_Lt_LOW"] is True:
        p=x["Bid1_low"]
        return [(p,v0)]
    a0=x["Turnover_DiffLen1S"]
    p1=math.floor(x["VWAP"])
    p2=p1+1
    v2=(a0-p1*v0)/1
    v1=v0-v2

    l=list()
    if v1>0:
        l.append((p1,v1))
    if v2>0:
        l.append((p2,v2))
    return l

df2["ExPriceTab"]=df2.apply(exchange_volume_guess2,axis=1)
df2


## 3.单边吃单 且 未穿盘

_df2=_df[~cond2]
cond3_up=((_df2["AskPrice1"]> _df2["AskPrice1_Lag1"])&
          (_df2["AskPrice1"]<=_df2["AskPrice5_Lag1"])&
          (_df2["BidPrice1"]>=_df2["BidPrice1_Lag1"]))

cond3_down=((_df2["BidPrice1"]< _df2["BidPrice1_Lag1"])&
            (_df2["BidPrice1"]>=_df2["BidPrice5_Lag1"])&
            (_df2["AskPrice1"]<=_df2["AskPrice1_Lag1"]))

df3_up=_df2[cond3_up]
df3_down=_df2[cond3_down]

usd_dict={
    "Ask1_up":"Bid1_low",
    "Bid1_low":"Ask1_up",
    "AskPrice1_Lag1":"BidPrice1_Lag1",
    "AskPrice2_Lag1":"BidPrice2_Lag1",
    "AskPrice3_Lag1":"BidPrice3_Lag1",
    "AskPrice4_Lag1":"BidPrice4_Lag1",
    "AskPrice5_Lag1":"BidPrice5_Lag1",

    "BidPrice1_Lag1":"AskPrice1_Lag1",
    "BidPrice2_Lag1":"AskPrice2_Lag1",
    "BidPrice3_Lag1":"AskPrice3_Lag1",
    "BidPrice4_Lag1":"AskPrice4_Lag1",
    "BidPrice5_Lag1":"AskPrice5_Lag1",

    "AskPrice1":"BidPrice1",
    "AskPrice2":"BidPrice2",
    "AskPrice3":"BidPrice3",
    "AskPrice4":"BidPrice4",
    "AskPrice5":"BidPrice5",

    "BidPrice1":"AskPrice1",
    "BidPrice2":"AskPrice2",
    "BidPrice3":"AskPrice3",
    "BidPrice4":"AskPrice4",
    "BidPrice5":"AskPrice5",
    "VWAP":"VWAP",
    "Turnover_DiffLen1S":"Turnover_DiffLen1S"}

usd_dict1 = {
    "AskVolume1_Lag1": "BidVolume1_Lag1", 
    "AskVolume2_Lag1": "BidVolume2_Lag1", 
    "AskVolume3_Lag1": "BidVolume3_Lag1", 
    "AskVolume4_Lag1": "BidVolume4_Lag1", 
    "AskVolume5_Lag1": "BidVolume5_Lag1",
    
    "BidVolume1_Lag1": "AskVolume1_Lag1", 
    "BidVolume2_Lag1": "AskVolume2_Lag1", 
    "BidVolume3_Lag1": "AskVolume3_Lag1", 
    "BidVolume4_Lag1": "AskVolume4_Lag1", 
    "BidVolume5_Lag1": "AskVolume5_Lag1", 

    "AskVolume1": "BidVolume1", 
    "AskVolume2": "BidVolume2", 
    "AskVolume3": "BidVolume3", 
    "AskVolume4": "BidVolume4", 
    "AskVolume5": "BidVolume5",
    
    "BidVolume1": "AskVolume1", 
    "BidVolume2": "AskVolume2", 
    "BidVolume3": "AskVolume3", 
    "BidVolume4": "AskVolume4", 
    "BidVolume5": "AskVolume5"


}


def upsdown(x):
    x1=x.copy()
    for i,j in usd_dict.items():
        x1[i]=-x[j]

    for i,j in usd_dict1.items():
        x1[i]=x[j]
        
    return x1

def p_inverse(x):
    return [(-i,j) for i,j in x]

def exchange_volume_guess3_up(x):
    v0=x["Volume_DiffLen1"]
    a=x["Turnover_DiffLen1S"]
    if v0==0:
        return []

    if x["VWAP"]>x["Ask1_up"]:
        return [(x["Ask1_up"],v0)]

    if x["VWAP"]<x["Bid1_low"]:
        return [(x["Bid1_low"],v0)]

    p1=x["AskPrice1_Lag1"]-1
    if (x["VWAP"]<p1):
        return [(p1,v0)]

    l=list()
    for i in range(1,6):
        _v,_p=x[f"AskVolume{i}_Lag1"],x[f"AskPrice{i}_Lag1"]
        
        if _p>x["Ask1_up"]:
            break
        
        if _p == x["Ask1_up"]:
            _v = _v - x["AskVolume1"]
            if _v > 0.5:
                l.append((_p-p1,_v))
            break
            
        l.append((_p-p1,_v))

    l1=list()
    a1=a-(x["AskPrice1_Lag1"]-1)*v0
    v1=v0
    for p,v in l:
        v2=min(v,v1)
        if v2*p>=a1:
            _v=round(a1/p)
            v1-=_v
            l1.append((p+p1,_v))
            a1 = 0
            break
        else:
            v1-=v2
            a1-=v2*p
            l1.append((p+p1,v2))
            
        if v1==0:
            break

    if v1==0:
        return l1

    left_avg = a1 / v1
    left_avg_int = math.floor(left_avg)
    left_avg_float = left_avg - left_avg_int
    if left_avg_float < 0.0001:
        l2 = [(p1 + left_avg_int, v1)]
    else:
        v = round(left_avg_float * v1)
        l2 = [(p1 + left_avg_int, v1 - v),
              (p1 + left_avg_int + 1, v)
              ]
    return l, merge_pv_tab(l2, l1)

def merge_pv_tab(t1, t2):
    t3 = list()
    i2=0
    l2=len(t2)

    for i1,(p1,v1) in enumerate(t1):
        v1_add=0
        while True:
            if i2>=l2:
                break
            p2,v2=t2[i2]
            if p2>p1:
                break
            if p2<p1:
                t3.append((p2,v2))
                i2+=1
                continue

            if p2==p1:
                v1_add=v2
                i2+=1
                break
        t3.append((p1,v1+v1_add))

    while True:
        if i2>=l2:
            break
        p2,v2=t2[i2]
        t3.append((p2,v2))
        i2+=1

    return t3



df3_up["ExPriceTab"]=df3_up.apply(exchange_volume_guess3_up,axis=1)
df3_up["ExPriceTab"].apply(sum_amt)-df3_up["Turnover_DiffLen1S"]

df3_down["ExPriceTab"]=df3_down.apply(lambda x:p_inverse(exchange_volume_guess3_up(upsdown(x))),axis=1)
df3_down["ExPriceTab"].apply(sum_amt)-df3_down["Turnover_DiffLen1S"]


###############################################################################################################
# (df3_down["VWAP"] - (df3_down["AskPrice1"] + df3_down["BidPrice1"]) / 2).abs().mean()                       #
# (df3_down["VWAP"] - (df3_down["AskPrice1_Lag1"] + df3_down["BidPrice1_Lag1"]) / 2).abs().mean()             #
#                                                                                                             #
#                                                                                                             #
# (df3_up["VWAP"] - (df3_up["AskPrice1"] + df3_up["BidPrice1"]) / 2).abs().mean()                             #
# (df3_up["VWAP"] - (df3_up["AskPrice1_Lag1"] + df3_up["BidPrice1_Lag1"]) / 2).abs().mean()                   #
# show_price_tab(_df3.iloc[10], "_Lag1").merge(show_price_tab(_df3.iloc[10]),                                 #
#                                              on="Price",                                                    #
#                                              how="outer").sort_values("Price", ascending=False).fillna( -1) #
###############################################################################################################



def price_tab(df):
    p1 = df["AskPrice5"]. max()
    p2 = df["BidPrice5"]. min()

    df_price = pd.DataFrame(columns=range(p2, p1 + 1), index=df.index)
df_price = df.apply(lambda x:{
        x["AskPrice5"]: x["AskVolume5"], 
        x["AskPrice4"]: x["AskVolume4"], 
        x["AskPrice3"]: x["AskVolume3"], 
        x["AskPrice2"]: x["AskVolume2"], 
        x["AskPrice1"]: x["AskVolume1"],
        x["BidPrice5"]: -x["BidVolume5"], 
        x["BidPrice4"]: -x["BidVolume4"], 
        x["BidPrice3"]: -x["BidVolume3"], 
        x["BidPrice2"]: -x["BidVolume2"], 
        x["BidPrice1"]: -x["BidVolume1"], 

}, axis=1)



df_price = pd.DataFrame(df_price.to_dict()).sort_index(ascending=False)
df_volume = df[["Volume_DiffLen1", "Turnover_DiffLen1S"]]
df_volume["Turnover_DiffLen1S"] = df["Turnover_DiffLen1S"] - df["Volume_DiffLen1"] * df["BidPrice1"]
df_price1 = pd.concat([df_volume. T, df_price])
df_price1.iloc[:, 50:60]
df_price1.iloc[:, 58:69]
df_price1.iloc[:, 68:80]

df_price

df["Volume"]
df["AskBidPrice_Intv_Lag1"]. value_counts()
df.b2(x="AskBidPrice_Intv_Lag1", y="Volume_DiffLen1", f=lambda x, y:{"cnt": y.shape[0], "mean": y.mean(), "median": y.median()})

df["AskBid1_minv"] = df.apply(lambda x:min(x["AskVolume1_Lag1"], x["BidVolume1_Lag1"]), axis=1)

df.b2(x="AskBid1_minv", y="Volume_DiffLen1", f=lambda x, y:{"cnt": y.shape[0], "mean": y.mean(), "median": y.median()})

df["AskBid1_minv"]








