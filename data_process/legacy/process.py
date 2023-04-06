import pandas as pd
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
import sql
import re
import gc
from sql import read_sql

#sql.client.execute("create database rb")
d1 = "20221201"
d2 = "20230324"
code = "rb"
tb_name = "rb.detail"

rb_data = get_feature_data(d1, d2, code)

for i in sorted(rb_data):
    i1 = rb_data[i]
    for j1, j in enumerate(["night", "day1", "day2", "day3"]):
        print((i, j1))        
        try:
            check_exists = read_sql(f"""select * from {tb_name}
             where TradingDay='{i[0]}'
            and Session='{j}' limit 1""")
            if check_exists.shape[0] > 0:
                print("Exists")
                continue
        except:
            pass

        df = i1[j]
        if df.shape[0] <= 100:
            continue
        
        if "level_0" in df:
            del df["level_0"]
        df = df[df.ncc("RT|RM")]
        df = df.reset_index(drop=True)
        df = append_feature_exch_detail(df)
        df = append_tick_detail(df)
        df = cc2(df, append_kline_ret_idx)
        df.tsq(tb_name, partition="TradingDay",
               orderby=["TradingDay", "ExchTimeOffsetUs"])
        gc.collect()
        


