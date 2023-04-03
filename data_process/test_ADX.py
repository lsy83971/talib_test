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
from timeseries_detail import func_info, total_func_info
from corr_analyze import xydata, beautify_excel
from jump_span import EMA as EMA1
from talib.abstract import * 


def append_DMIIdx(data):
    res = dict()
    for i in [14, 30, 60, 120, 180, 240, 360, 480, 600,
              720, 960, 1200, 1800, 2400, 3600
              ]:
        print(i)
        res["ADX_" + str(i)] = ADX(data, timeperiod=i)
        res["ATR_" + str(i)] = ATR(data, timeperiod=i)
        res["PLUS_DM_" + str(i)] = PLUS_DM(data, timeperiod=i)
        res["MINUS_DM_" + str(i)] = MINUS_DM(data, timeperiod=i)
        res["PLUS_DI_" + str(i)] = PLUS_DI(data, timeperiod=i)
        res["MINUS_DI_" + str(i)] = MINUS_DI(data, timeperiod=i)
        
    res_df = pd.DataFrame(res)
    res_df.columns = "TXDMI_" + res_df.columns
    return res_df



if __name__ == "__main__":
    df = pd.read_pickle("./test_data/kline.pkl")
    df = cc2(df, append_DMIIdx)
    df = xydata(df)
    df.cross_corr()
    df.daywise_corr()
    df.to_excel(f"./output/DMI_tick.xlsx", append_info={"function_info": func_info})














