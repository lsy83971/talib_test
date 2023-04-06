import pandas as pd
import numpy as np
import math
from append_df import cc1, cc2

## 1. basic operators of pv-dict
def sum_amt(x):
    a = 0
    for i, j in x.items():
        a+=i * j
    return a

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


## 3. calculate pv-dict

def append_lag(df):
    res = dict()
    df['Turnover_DiffLen1S'] = df['Turnover_DiffLen1'] / 10    
    res["Volume_DiffLen1_Lag1"] = df["Volume_DiffLen1"]. shift(1)
    res["Volume_DiffLen1_Lag2"] = df["Volume_DiffLen1"]. shift(2)
    res["Volume_DiffLen1_Lag3"] = df["Volume_DiffLen1"]. shift(3)
    res["Volume_DiffLen1_Lag4"] = df["Volume_DiffLen1"]. shift(4)
    res["Volume_DiffLen1_Lag5"] = df["Volume_DiffLen1"]. shift(5)
    res["Turnover_DiffLen1S_Lag1"] = df["Turnover_DiffLen1S"]. shift(1)
    res["Turnover_DiffLen1S_Lag2"] = df["Turnover_DiffLen1S"]. shift(2)
    res["Turnover_DiffLen1S_Lag3"] = df["Turnover_DiffLen1S"]. shift(3)
    res["Turnover_DiffLen1S_Lag4"] = df["Turnover_DiffLen1S"]. shift(4)
    res["Turnover_DiffLen1S_Lag5"] = df["Turnover_DiffLen1S"]. shift(5)
    return pd.DataFrame(res)

def append_pv_dict_lag(df):
    res = dict()
    res["ask_dict_lag1"] = df["ask_dict"]. shift(1)
    res["ask_dict_cumsum_lag1"] = df["ask_dict_cumsum"]. shift(1)
    
    res["bid_dict_lag1"] = df["bid_dict"]. shift(1)
    res["bid_dict_cumsum_lag1"] = df["bid_dict_cumsum"]. shift(1)        
    return pd.DataFrame(res)

def calc_pv_dict(x):
    res = dict()
    res["ask_dict"] = {x[f"AskPrice{i}"]: x[f"AskVolume{i}"] for i in range(1, 6)}
    res["bid_dict"] = {x[f"BidPrice{i}"]: x[f"BidVolume{i}"] for i in range(1, 6)}

    ask_dict_price = sorted(res["ask_dict"])
    k = 0
    res["ask_dict_cumsum"] = dict()
    for i in range(ask_dict_price[0], ask_dict_price[ - 1] + 1):
        k += res["ask_dict"]. get(i, 0)
        res["ask_dict_cumsum"][i] = k

    bid_dict_price = sorted(res["bid_dict"])
    k = 0
    res["bid_dict_cumsum"] = dict()
    for i in range(bid_dict_price[ - 1], bid_dict_price[0] - 1, -1):
        k += res["bid_dict"]. get(i, 0)
        res["bid_dict_cumsum"][i] = k
        
    return res



## 4.calculate exch detail
## **************** 一 逻辑疏理 ***********************
###############################
#       Lag1    now           #
# 4229  -1000  -1000          #
# 4228  -100   -100           #
# 4227  +1000  +500           #
# 4226  +500   +500           #
# volume 500, VWAP4227        #
###############################
# 上述情形为例 为满足量价关系 最合理情形为4227卖出500:

############################################
#        Lag1               now            #
# 4229  -1000              -1000           #
# 4228  -100               -100            #
# 4227  +1000  sell 500=>  +500            #
# 4226  +500               +500            #
############################################

# 假设一种情况4226成交250 4228成交250 同样可以满足量价关系 但是成交方式就会非常极端:
##############################################################################
#        Lag1     step1      now  step2      now     step3         now       #
# 4229  -1000               -1000           -1000                 -1000      #
# 4228  -100                -100            -100  buy250 sell250  -100       #
# 4227  +1000 cancel 1000=> 0               0     buy500 =>       +500       #
# 4226  +500                +500 sell250 => +250  buy250 =>       +500       #
##############################################################################
# 此模式不合理之处有二
# 1. 撤/挂单总额过大
# 2. step1/2/3 必须严格满足先后次序

## **************** 二 制定惩罚函数 ***********************
# 总结完以上规律 
# 我们后续目标是设计一个惩罚函数 对于符合量价关系的每种交易定义惩罚函数
# 寻找最合理的交易模式 即最小化惩罚函数

# 惩罚函数=挂/撤单总量+过路费
# 1.挂/撤单总量 即所有档位 挂/撤总量求和
# 2.过路费(以上述为例)
# 在step2，成交价击穿4227来到4226，sell成交250手。
# 而击穿当前价卖出成交的这250手，必须在step3-4227挂单500手之前成交。
# 基于以上考虑 设计如下惩罚项：
# 交易价格想向下/向上击穿某价格，必须要交过路费。
# 过路费大小为step3买入/卖出量，在当前例子下过路费为500。
# 当前例子下
# 前者 obj_func=500
# 后者 obj_func=(1000+250*2+250*2+500)+500=3000

## ************** 三 惩罚函数优化方法 *********************
# 1. 过路费只增不减
# 2. 红利部分/普通部分/过路费部分

def calc_moat_price(x):
    res = dict()
    moat_thres = x["Volume_DiffLen1"] / 2 + 50
    
    A1P_max = int(max(x["AskPrice1_Lag1"], x["AskPrice1"]))
    A5P_min = int(min(x["AskPrice5_Lag1"], x["AskPrice5"]))
    moat_ask = min(A1P_max, A5P_min)
    for i in range(moat_ask, min(moat_ask + 3, A5P_min + 1)):
        moat_ask = i
        v0 = x["ask_dict_cumsum"]. get(i, 0)
        v1 = x["ask_dict_cumsum_lag1"]. get(i, 0)
        if (v0 > moat_thres) and (v1 > moat_thres):
            break
    res["A1P_max"] = A1P_max
    res["moat_ask"] = moat_ask

    B1P_min = int(min(x["BidPrice1_Lag1"], x["BidPrice1"]))
    B5P_max = int(max(x["BidPrice5_Lag1"], x["BidPrice5"]))
    moat_bid = max(B1P_min, B5P_max)
    for i in range(moat_bid, max(moat_bid - 3, B5P_max-1), -1):
        moat_bid = i
        v0 = x["bid_dict_cumsum"]. get(i, 0)
        v1 = x["bid_dict_cumsum_lag1"]. get(i, 0)
        if (v0 > moat_thres) and (v1 > moat_thres):
            break
        #if v0 + v1 > moat_thres:
        #    break
    res["B1P_min"] = B1P_min
    res["moat_bid"] = moat_bid

    return res

def calc_raw_exch_tab(v, t):
    if t == 0:
        return dict()
    if pd.isnull(v):
        return dict()
    p1 = math.floor(t / v)
    p2 = p1 + 1
    v2 = (a - p1 * v)
    v1 = v - v2
    return {p1: v1,p2: v2}

def calc_exch_detail(x):
    res = dict()
    if pd.isnull(x["AskPrice1_Lag1"]):
        return {"exch_detail": dict()}
    v = x["Volume_DiffLen1"]
    t = x["Turnover_DiffLen1S"]
    VWAP = x["VWAP"]
    moat_info = calc_moat_price(x)
    m1 = moat_info["moat_bid"]
    m2 = moat_info["moat_ask"]
    
    if VWAP > m2:
        return {"exch_detail":{m2: v}}
    if VWAP < m1:
        return {"exch_detail":{m1: v}}

    ask_v1 = cut_dict(pos_dict(sub_dict(x["ask_dict_lag1"], x["ask_dict"])), m1, m2)
    bid_v1 = cut_dict(pos_dict(sub_dict(x["bid_dict_lag1"], x["bid_dict"])), m1, m2)

    ask_v1_cumsum = dict()
    ask_t1_cumsum = dict()    
    v1 = 0
    v2 = 0
    for i in range(m1, m2 + 1):
        v1 += ask_v1.get(i, 0)
        v2 += x["ask_dict_lag1"]. get(i, 0)* i        
        ask_v1_cumsum[i] = v1
        ask_t1_cumsum[i] = v2
        
    bid_v1_cumsum = dict()
    bid_t1_cumsum = dict()    
    v1 = 0
    v2 = 0
    for i in range(m2, m1 - 1, -1):
        v1 += bid_v1.get(i, 0)
        v2 += x["bid_dict_lag1"]. get(i, 0)* i
        bid_v1_cumsum[i] = v1
        bid_t1_cumsum[i] = v2

    obj_score = dict()
    obj_res = dict()
    obj_detail = dict()
    for l1 in range(m1, m2 + 1):
        for l2 in range(l1, m2 + 1):
            #print(l1, l2)
            pun1a = x["ask_dict_cumsum_lag1"].get(l1 - 1, 0)
            pun1b = x["bid_dict_cumsum_lag1"].get(l2 + 1, 0)
            pun2a = x["ask_dict_cumsum"].get(l1 - 1, 0)
            pun2b = x["bid_dict_cumsum"].get(l2 + 1, 0)

            pun_out = pun1a + pun1b + pun2a + pun2b

            v0 = v
            t0 = t
            
            if l1 == l2:
                if t0 != v0 * l1:
                    continue
                p1 = ask_v1.get(l1, 0) + bid_v1.get(l1, 0)
                prz = min(p1, t0)
                score = prz - pun_out
                obj_res[(l1, l2)] = {l1: v0}
                obj_score[(l1, l2)] = score
                continue
            
            if (t0 >= l2 * v0) or (t0 <= l1 * v0):
                continue
            
            if l2 == l1 + 1:
                v2 = (t0 - v0 * l1)
                v1 = v0 - v2
                p1 = min(v1, ask_v1.get(l1, 0) + bid_v1.get(l1, 0))
                p2 = min(v2, ask_v1.get(l2, 0) + bid_v1.get(l2, 0))         
                prz = p1 + p2
                score = prz - pun_out
                obj_res[(l1, l2)] = {l1: v1, l2: v2}
                obj_score[(l1, l2)] = score
                continue

            p_add = bid_v1_cumsum.get(l1 + 1, 0) - bid_v1_cumsum.get(l2, 0) + \
                    ask_v1_cumsum.get(l2 - 1, 0) - ask_v1_cumsum.get(l1, 0)
            t_add = bid_t1_cumsum.get(l1 + 1, 0) - bid_t1_cumsum.get(l2, 0) + \
                    ask_t1_cumsum.get(l2 - 1, 0) - ask_t1_cumsum.get(l1, 0)
            v_add = x["bid_dict_cumsum_lag1"].get(l1 + 1, 0) - \
                    x["bid_dict_cumsum_lag1"].get(l2, 0) + \
                    x["ask_dict_cumsum_lag1"].get(l2 - 1, 0) - \
                    x["ask_dict_cumsum_lag1"].get(l1, 0)
            pun_inner = x["bid_dict_cumsum"].get(l1 + 1, 0) - \
                x["bid_dict_cumsum"].get(l2, 0) + \
                x["ask_dict_cumsum"].get(l2 - 1, 0) - \
                x["ask_dict_cumsum"].get(l1, 0)            
            
            v0 = v0 - v_add
            t0 = t0 - t_add
            if v0 <= 0:
                continue

            if (t0 >= l2 * v0) or (t0 <= l1 * v0):
                continue

            v1 = (l2 * v0 - t0) // (l2 - l1)
            v2 = v0 - v1

            p1 = min(v1, ask_v1.get(l1, 0) + bid_v1.get(l1, 0))
            p2 = min(v2, ask_v1.get(l2, 0) + bid_v1.get(l2, 0))

            prz = p1 + p2 + p_add
            #print(prz, pun_inner, pun_out)
            score = prz - pun_out - pun_inner

            tmp_dict = {i:x["ask_dict_lag1"]. get(i, 0) + x["bid_dict_lag1"]. get(i, 0) \
                        for i in range(l1 + 1, l2)}
            tmp_dict[l1] = v1
            tmp_dict[l2] = v2
            x["bid_dict_lag1"]
            obj_res[(l1, l2)] = tmp_dict
            obj_score[(l1, l2)] = score
            obj_detail[(l1, l2)] = {"pun_inner": pun_inner, "pun_out": pun_out, "prz": prz}
            continue
    score_max = max(obj_score.values())
    idx_max = [i for i, j in obj_score.items() if j == score_max][0]
    res["exch_detail"] = obj_res[idx_max]
    res.update(moat_info)
    return res

def append_feature_exch_detail(df):
    df = cc2(df, append_lag)
    df = cc1(df, calc_pv_dict)
    df = cc2(df, append_pv_dict_lag)
    df = cc1(df, calc_exch_detail)
    return df

if __name__ == "__main__":
    pass







    

    
    

    
    
    
