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
from sql import rsq, read_sql, get_kline
import warnings
warnings.simplefilter('ignore')
import re
import gc
from timeseries_detail import func_info
from corr_analyze import xydata, beautify_excel

data = get_kline("rb.detail")
data = cc2(data, append_crossday_return)
Ry = data.cc("RT|RM").ncc("ORT|ORM").tolist()
data = cc2(data, append_techIdxBasic)
data = cc2(data, append_MAIdx)

data[data.cc("^OR").tolist() + ["open", "close", "high", "low", "volume"] + ["TradingDay", "time", "Session"]]. to_pickle("./test_data/kline.pkl")

df = xydata(data)
df.erase_kline_tick_data()
df.cross_corr()
df.daywise_corr()
df.to_excel(f"./output/basic_tick.xlsx", append_info={"function_info": func_info})

for period in [1, 5, 10, 30, 60]:
    data = get_kline("rb.detail", period)
    data = cc2(data, append_crossday_return)
    data = cc2(data, append_techIdxBasic)
    data = cc2(data, append_MAIdx)

    df = xydata(data)
    df.cross_corr()
    df.daywise_corr()
    df.to_excel(f"./output/basic_{period}s.xlsx", append_info={"function_info": func_info})
