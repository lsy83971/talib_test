import pandas as pd
import numpy as np
import math
from append_df import cc1, cc2
import talib
from talib.abstract import *
import sys
sys.path.append("/home/lishiyu/Project/bin_tools")
from bins import binning, bins_simple_mean
from pvdict_utils import *
from exch_detail import append_feature_exch_detail


def fna(self):
    return self.fillna(method="ffill").fillna(method="bfill")
pd.Series.fna = fna

def append_kline_ret_idx(df):
    #df = cc2(df,add_line_idx)
    price_chanage_min = 1
    res = dict()
    
    res["tick_high"] = df["exch_detail_max"]
    res["tick_low"] = df["exch_detail_min"]
    res["tick_high"] = res["tick_high"]. fna()
    res["tick_low"] = res["tick_low"]. fna()

    res["VWAP_end"] = (df["Turnover_DiffLen1S"]. fillna(0) + \
                       df["Turnover_DiffLen1S"]. shift( -1).fillna(0)) / \
                       (df["Volume_DiffLen1"]. fillna(0) + df["Volume_DiffLen1"]. shift( -1).fillna(0))
    res["VWAP_front"] = res["VWAP_end"]. shift(1)

    res["VWAP_front"] = res["VWAP_front"]. fna()
    res["VWAP_end"] = res["VWAP_end"]. fna()
    res["VWAP_front"] = round(res["VWAP_front"] / price_chanage_min)*price_chanage_min
    res["VWAP_end"] = round(res["VWAP_end"] / price_chanage_min)*price_chanage_min
    res["VWAP_end"] = np.maximum(np.minimum(res["VWAP_end"], res["tick_high"]), res["tick_low"])
    res["VWAP_front"] = np.maximum(np.minimum(res["VWAP_front"], res["tick_high"]), res["tick_low"])

    res["tick_open"] = res["VWAP_front"]
    res["tick_close"] = res["VWAP_end"]

    res["RT20"] = (res["tick_close"]. shift( -20) - res["tick_close"])
    res["RT40"] = (res["tick_close"]. shift( -40) - res["tick_close"])
    res["RT60"] = (res["tick_close"]. shift( -60) - res["tick_close"])
    res["RM1"] = (res["tick_close"]. shift( -120) - res["tick_close"])
    res["RM3"] = (res["tick_close"]. shift( -360) - res["tick_close"])        
    res["RM5"] = (res["tick_close"]. shift( -600) - res["tick_close"])
    res["RM10"] = (res["tick_close"]. shift( -1200) - res["tick_close"])
    res["RM15"] = (res["tick_close"]. shift( -1800) - res["tick_close"])
    res["RM20"] = (res["tick_close"]. shift( -2400) - res["tick_close"])
    res["RM30"] = (res["tick_close"]. shift( -3600) - res["tick_close"])
    return res

### type 0 group by date
### type 1 group by (date (morning, afternoon,night))
### type 2 group by (date (day, night))

from get_feature_data import trading_time_interval_dict
begin_tick_dict = {i:j[0] for i, j in trading_time_interval_dict.items()}
end_tick_dict = {i:j[1] for i, j in trading_time_interval_dict.items()}

begin_tick_dict1 = begin_tick_dict.copy()
end_tick_dict1 = end_tick_dict.copy()
begin_tick_dict1["day2"] = begin_tick_dict["day1"]
end_tick_dict1["day1"] = end_tick_dict["day2"]

begin_tick_dict2 = begin_tick_dict.copy()
end_tick_dict2 = end_tick_dict.copy()
begin_tick_dict2["day3"] = begin_tick_dict["day1"]
begin_tick_dict2["day2"] = begin_tick_dict["day1"]
end_tick_dict2["day2"] = end_tick_dict["day3"]
end_tick_dict2["day1"] = end_tick_dict["day3"]

begin_tick_dict0 = begin_tick_dict.copy()
end_tick_dict0 = end_tick_dict.copy()
begin_tick_dict0["day3"] = begin_tick_dict["night"]
begin_tick_dict0["day2"] = begin_tick_dict["night"]
begin_tick_dict0["day1"] = begin_tick_dict["night"]
end_tick_dict0["night"] = end_tick_dict["day3"]
end_tick_dict0["day2"] = end_tick_dict["day3"]
end_tick_dict0["day1"] = end_tick_dict["day3"]

def append_crossday_return(df):
    #df = cc2(df,append_crossday_return)    
    res = dict()
    
    res["t_begin"] = df["Session"]. apply(lambda x:begin_tick_dict.get(x))
    res["t_end"] = df["Session"]. apply(lambda x:end_tick_dict.get(x))
    res["from_begin"] = df["time"] - res["t_begin"]
    res["to_end"] = res["t_end"] - df["time"]

    res["t_begin1"] = df["Session"]. apply(lambda x:begin_tick_dict1.get(x))
    res["t_end1"] = df["Session"]. apply(lambda x:end_tick_dict1.get(x))
    res["from_begin1"] = df["time"] - res["t_begin1"]
    res["to_end1"] = res["t_end1"] - df["time"]

    res["t_begin2"] = df["Session"]. apply(lambda x:begin_tick_dict2.get(x))
    res["t_end2"] = df["Session"]. apply(lambda x:end_tick_dict2.get(x))
    res["from_begin2"] = df["time"] - res["t_begin2"]
    res["to_end2"] = res["t_end2"] - df["time"]

    res["t_begin0"] = df["Session"]. apply(lambda x:begin_tick_dict0.get(x))
    res["t_end0"] = df["Session"]. apply(lambda x:end_tick_dict0.get(x))
    res["from_begin0"] = df["time"] - res["t_begin0"]
    res["to_end0"] = res["t_end0"] - df["time"]

    for i in [1, 3, 5, 10, 20, 30, 60]:
        res[f"ORT{i}"] = (df["close"]. shift( -i) - df["close"])

    for i in [1, 3, 5, 10, 20, 30]:
        res[f"ORM{i}"] = (df["close"]. shift( -i * 120) - df["close"])
        
    return res

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
    res = dict()
    for i in [10, 30, 60, 120, 180, 240, 360, 480, 600,
              720, 960, 1200, 1800, 2400, 3600
              ]:
        res["sma_" + str(i)] = SMA(data, timeperiod=i) - data["close"]
        res["ema_" + str(i)] = EMA(data, timeperiod=i) - data["close"]
        
    res_df = pd.DataFrame(res)
    res_df.columns = "TXMA_" + res_df.columns
    return res_df

