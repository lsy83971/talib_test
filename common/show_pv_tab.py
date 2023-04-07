import pandas as pd
import numpy as np
import math
from importlib import reload
import common.dict_operator
reload(common.dict_operator)
from common.dict_operator import DPop, DSub, DSumMo
from local_sql import read_sql


def get_lag_df(df, i, lag=10):
    if i >= 0:
        tmpdf = df.iloc[max(0, i - lag):(i + 1)]
    if i < 0:
        tmpdf = df.iloc[(i - lag):(i + 1)]
    return tmpdf

def show_pv_tab(df, i, lag=10):
    tmpdf = get_lag_df(df, i, lag)
    info_pv = pd.DataFrame(tmpdf.apply(lambda x:DPop(DSub(x["D_ask"], x["D_bid"])), axis=1).to_dict()).sort_index(ascending=False).fillna("")
    info_pv.loc["***"] = ""

    tmpdf["VWAP"] = (tmpdf["amt"] / tmpdf["vol"]).round(2) - tmpdf["D_bid"]. apply(lambda x:max(x.keys()))
    tmpdf["vol"] = tmpdf["vol"]. astype(int)
    info_av = tmpdf[["vol", "VWAP"]]. T
    info_av.loc["***"] = ""
    info = pd.concat([info_pv, info_av])
    return info

def show_exch_tab(df, i, lag=10):
    tmpdf = get_lag_df(df, i, lag)
    info_exch = pd.DataFrame(tmpdf.apply(lambda x:DPop(x["D_exch"]), axis=1).to_dict()).sort_index(ascending=False).fillna("")
    info_exch.loc["***"] = ""
    return info_exch

def show3(df, i, lag=10):
    return pd.concat([show_pv_tab(df, i, lag=lag), show_exch_tab(df, i, lag=lag)])


df = read_sql("""select
D_exch_moment
,AP1
,AV1
,time
,date
,ask_bound
,bid_bound
,Session
,time
,D_exch
,vol
,amt
,D_ask
,D_bid
,D_ask_last
,D_bid_last
from rb.tickdata
where date='2022-12-19'
order by time
""")

(df["D_exch"]. apply(DSumMo) - df["amt"]).value_counts().sort_index()
df["D_exch"]. apply(len).value_counts().sort_index()

sb = df[(df["D_exch"]. apply(DSumMo) - df["amt"]) > 0]. index
sb
gg["D_ask"]

# df.iloc[14399]["Session"]
# df.iloc[14398]["Session"]
df.iloc[4574]
show3(df, sb[0] + 5, 10)
show3(df, 5, 10)
show3(df, sb[0] + 55, 60)
show3(df, 4600, 30)
show3(df, 4740, 30)
show3(df, 14399, 5)
df.iloc[4574]["D_ask_last"]
df.iloc[4574]["D_ask_last"]
i = 10
show_pv_tab(df, -3, 5)
df["time"]
df.iloc[0]

pd.set_option("display.max_columns", 200)


df.iloc[0]["D_bid"]
