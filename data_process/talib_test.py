import pandas as pd
import numpy as np
import math
import talib
from talib.abstract import *

TX_function = talib.get_functions()
total_func_info = pd.DataFrame({i: eval(i).info for i in TX_function}).T
total_func_info["output_num"] = total_func_info["output_names"]. apply(len)
cond1 = total_func_info["output_num"] == 1
cond2 = total_func_info["output_num"] > 1
total_func_info.loc[cond1, "output_names"] = total_func_info.loc[cond1, "name"]. apply(lambda x:[x])
total_func_info.loc[cond2, "output_names"] = total_func_info.loc[cond2]. apply(
    lambda x:[x["name"] + "_" + i for i in x["output_names"]], axis=1)

func_info = total_func_info.loc[total_func_info["group"]. ncc("Math|Price Transform|Statistic Functions").index]
func_info = func_info.loc[func_info["name"]. ncc("MAVP").index]
single_func = func_info[func_info["output_num"] == 1]
multi_func = func_info[func_info["output_num"] > 1]

def append_techIdxBasic(data):
    """
    append talib basic idx
    use default parameters
    """
    dfs = list()
    res = dict()
    for i in single_func.index:
        if i == "MAVP":
            continue
        res[i] = eval(i)(data)
    dfs.append(pd.DataFrame(res))    

    for i in multi_func.index:
        tmp = eval(i)(data)
        tmp.columns = i + "_" + tmp.columns
        dfs.append(tmp)
        
    res_df = pd.concat(dfs, axis=1)
    for idx in [j for i in func_info[func_info["group"] == "Overlap Studies"]["output_names"] for j in i]:
        if idx not in res_df.columns:
            continue
        res_df[idx] = res_df[idx] - data["close"]

    res_df.columns = "TXB_" + res_df.columns
    return res_df

def append_MAIdx(data):
    """
    change timeperiod with sma/ema
    """
    res = dict()
    for i in [10, 30, 60, 120, 180, 240, 360, 480, 600,
              720, 960, 1200, 1800, 2400, 3600
              ]:
        res["sma_" + str(i)] = SMA(data, timeperiod=i) - data["close"]
        res["ema_" + str(i)] = EMA(data, timeperiod=i) - data["close"]
        
    res_df = pd.DataFrame(res)
    res_df.columns = "TXMA_" + res_df.columns
    return res_df

