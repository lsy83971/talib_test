import pandas as pd
pd.set_option("display.max_rows", 500)
import numpy as np
import math
from append_df import cc1, cc2
import sys
sys.path.append("/home/lishiyu/Project/bin_tools")
from bins import binning, bins_simple_mean
from pvdict_utils import *
from exch_detail import append_feature_exch_detail
from tick_detail import append_tick_detail
from get_feature_data import get_feature_data
from timeseries_detail import *
from clickhouse_driver import Client
from pandas.api.types import infer_dtype
from sql import rsq
import re
import gc

import sys
sys.path.append("../")
from explore.backtest import ret_ts

sql = """select
TradingDay,
ExchTimeOffsetUs,
Session,
tick_open,
tick_close,
tick_high,
tick_low,
Volume_DiffLen1,
Turnover_DiffLen1S,
MX_exch_detail,
RT20,
RT40,
RT60,
RM1,
RM3,
RM5,
RM10,
RM15,
RM20,
RM30
from rb.detail
where TradingDay>='2023-02-01'
and TradingDay<='2023-03-18'
order by TradingDay, ExchTimeOffsetUs
"""

data = rsq(sql)
data = cc2(data, append_crossday_return)
data.r1({"tick_open": "open",
         "tick_close": "close",
         "tick_high": "high",
         "tick_low": "low",
         "Volume_DiffLen1": "volume",          
         })
Ry = data.cc("RT|RM").ncc("ORT|ORM").tolist()
data = cc2(data, append_techIdxBasic)
data = cc2(data, append_MAIdx)

erase_y = {
    "ORT20":20, 
    "ORT40":40, 
    "ORT60":60, 
    "ORM1":120, 
    "ORM3":360, 
    "ORM5":600, 
    "ORM10":1200, 
    "ORM15":1800, 
    "ORM20":2400, 
    "ORM30":3600, 
    "ORM40":4800, 
    "ORM50":6000, 
    "ORM60":7200, 
}

erase_x = {
    "TX_sma_10":10, 
    "TX_sma_30":30, 
    "TX_sma_60":60, 
    "TX_sma_120":120, 
    "TX_sma_180":180, 
    "TX_sma_240":240, 
    "TX_sma_360":360, 
    "TX_sma_480":480,
    "TX_sma_600": 600, 
    "TX_sma_720": 720, 
    "TX_sma_960": 960, 
    "TX_sma_1200": 1200, 
    "TX_sma_1800": 1800, 
    "TX_sma_2400": 2400, 
    "TX_sma_3600": 3600, 
    
    "TX_ema_10":60, 
    "TX_ema_30":60, 
    "TX_ema_60":120, 
    "TX_ema_120":240, 
    "TX_ema_180":360, 
    "TX_ema_240":480, 
    "TX_ema_360":720, 
    "TX_ema_480":960,
    "TX_ema_600": 600, 
    "TX_ema_720": 720, 
    "TX_ema_960": 960, 
    "TX_ema_1200": 1200, 
    "TX_ema_1800": 1800, 
    "TX_ema_2400": 2400, 
    "TX_ema_3600": 3600, 
}

xidx = data.cc("^TX")
yidx = data.cc("^ORM|^ORT")

for i in xidx:
    erase_period = erase_x.get(i, 60)
    data.loc[data["from_begin0"] <= erase_period / 2, i] = None

for i in yidx:
    erase_period = erase_y.get(i, 60)
    data.loc[data["to_end0"] <= erase_period / 2, i] = None

    
dfx = data[xidx]
dfy = data[yidx]


def cross_mean(dfx, dfy):
    dfx_v = (~dfx. isnull()).astype(np.float64)
    dfy_v = (~dfy. isnull()).astype(np.float64)
    dfx.fillna(0, inplace=True)
    dfy.fillna(0, inplace=True)
    
    corr_vcnt = (dfy_v.T)@(dfx_v)
    corr_xsum = (dfy_v.T)@(dfx)
    corr_ysum = (dfy.T)@(dfx_v)

    corr_xmean = (corr_xsum / corr_vcnt)
    corr_ymean = (corr_ysum / corr_vcnt)
    return corr_xmean, corr_ymean


def cross_corr(dfx, dfy):
    dfx_v = (~dfx. isnull()).astype(np.float64)
    dfy_v = (~dfy. isnull()).astype(np.float64)
    dfx.fillna(0, inplace=True)
    dfy.fillna(0, inplace=True)

    corr_crosssum = dfy.T@dfx
    corr_vcnt = (dfy_v.T)@(dfx_v)
    corr_xsum = (dfy_v.T)@(dfx)
    corr_ysum = (dfy.T)@(dfx_v)

    corr_xmean = (corr_xsum / corr_vcnt)
    corr_ymean = (corr_ysum / corr_vcnt)

    corr_xsqr = dfy_v.T@(dfx ** 2)
    corr_ysqr = (dfy.T ** 2)@(dfx_v)

    corr_xsqr_N = corr_xsqr - (corr_xmean ** 2)*(corr_vcnt)
    corr_ysqr_N = corr_ysqr - (corr_ymean ** 2)*(corr_vcnt)
    corr_crosssum_N = corr_crosssum - (corr_xmean * corr_ymean * corr_vcnt)
    corr_xy = corr_crosssum_N / ((corr_ysqr_N * corr_xsqr_N)** (1 / 2))
    return corr_xy

corr_xy = cross_corr(dfx, dfy)

if __name__ == "__main__":
    corr_xy.to_excel("./output/xy_correlation.xlsx")
    from timeseries_detail import total_func_info
    total_func_info.to_excel("./output/idx_info.xlsx")





