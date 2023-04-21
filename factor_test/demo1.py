import numpy as np
import sys
sys.path.append("/home/lishiyu/talib_test/data_process/")
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",2000)
pd.set_option("display.max_columns",2000)
pd.options.display.float_format = '{:.3f}'.format
import warnings
from cython_func.split_calc import split_idx, split_cumsum, split_cummax
from IPython.display import display
warnings.filterwarnings("ignore")
from common.show_pv_tab import show_pv,show3, show2
from data_process.local_sql import read_sql, exch_detail, client
from data_process.kline import table_kline_period
from data_process.corr_analyze import cross_corr_rank

cs_ls = ["D_ask_exch_old",
         "D_bid_exch_old", "D_exch_up_new", "D_exch_down_new", "D_exch_mid",
         "D_bid_add",
         "D_bid_cancel",
         "D_ask_add",
         "D_ask_cancel",              
         ]
         
period = [5, 10, 20, 30]
ls = [f"{i}_cumsum_{j}" for i in cs_ls for j in [5, 30]]

df = read_sql(f"""
select
b.MRT5 as MRT5,
b.MRT10 as MRT10,
b.MRM1 as MRM1,
c.D_exch as D_exch,
c.D_ask as D_ask,
c.D_bid as D_bid,
c.AP1 as AP1,
c.BP1 as BP1,
c.vol as vol,
c.amt as amt,
{','.join(
[f"arraySum(mapValues(clipMapLeft(a.{i},c.AP1))) as {i}_abvAP1" for i in ls]
)},
{','.join(
[f"arraySum(mapValues(clipMapRight(a.{i},c.BP1))) as {i}_blwBP1" for i in ls]
)},
{','.join(
[f"arraySum(mapValues(a.{i})) as {i}" for i in ls]
)},
{','.join(
[f"a.{i}[c.AP1] as {i}_AP1" for i in ls]
)},
{','.join(
[f"a.{i}[c.BP1] as {i}_BP1" for i in ls]
)}


from rb.tickdata as c
inner join rb.kline_1 as b
on b.date=c.date and b.time=c.time
inner join rb.DSMA as a
on a.date=c.date and a.time=c.time
order by a.date,a.time
""")

idy = df.cc("^MR[TM]").tolist()[:: -1]

df["test1"] = df["D_ask_exch_old_cumsum_30_abvAP1"] + df["D_exch_up_new_cumsum_30_abvAP1"]
df["test2"] = df["D_bid_exch_old_cumsum_30_blwBP1"] + df["D_exch_down_new_cumsum_30_blwBP1"]
df["test3"] = (df["test1"] + 1) / (df["test1"] + df["test2"] + 2)
df[["test3"] + idy]. corr("spearman")

df["test1"] = df["D_ask_exch_old_cumsum_30_abvAP1"]
df["test2"] = df["D_bid_exch_old_cumsum_30_blwBP1"]
df["test3"] = (df["test1"] + 1) / (df["test1"] + df["test2"] + 2)
df[["test3"] + idy]. corr("spearman")

df["test1"] = df["D_ask_exch_old_cumsum_30"] + df["D_exch_up_new_cumsum_30"]
df["test2"] = df["D_bid_exch_old_cumsum_30"] + df["D_exch_down_new_cumsum_30"]
df["test3"] = (df["test1"] + 1) / (df["test1"] + df["test2"] + 2)
df[["test3"] + idy]. corr("spearman")

df["test1"] = df["D_ask_exch_old_cumsum_30_abvAP1"]
df["test2"] = df["D_bid_exch_old_cumsum_30"]
df["test3"] = (df["test1"] + 1) / (df["test1"] + df["test2"] + 2)
df["test3"] = (df["test1"] + 1) / (df["test2"] + 2)
df[["test3"] + idy]. corr("spearman")






df["test1"] = df["D_ask_add_cumsum_30_abvAP1"] + 1
df["test2"] = df["D_ask_add_cumsum_30_abvAP1"] + df["D_ask_add_cumsum_30_blwBP1"] + 2
df["test3"] = df["test1"] / df["test2"]
df["test3"]. mean()
cross_corr_rank(df["test3"], df[idy])

df["test1_1"] = df["D_bid_add_cumsum_30_abvAP1"] + 1
df["test2_1"] = df["D_bid_add_cumsum_30_abvAP1"] + df["D_bid_add_cumsum_30_blwBP1"] + 2
df["test3_1"] = df["test1_1"] / df["test2_1"]
df["test3_1"]. mean()
cross_corr_rank(df["test3"], df[idy])

df["test1"] = df["D_ask_add_cumsum_30_abvAP1"] + df["D_bid_add_cumsum_30_abvAP1"] + 1
df["test2"] = df["D_ask_add_cumsum_30"] + df["D_bid_add_cumsum_30"] + 2
df["test3"] = df["test1"] / df["test2"]
df["test3"]. mean()
cross_corr_rank(df["test3"], df[idy])

df["test1"] = df["D_ask_add_cumsum_5_abvAP1"] + df["D_bid_add_cumsum_5_abvAP1"] + 1
df["test2"] = df["D_ask_add_cumsum_5"] + df["D_bid_add_cumsum_5"] + 2
df["test3"] = df["test1"] / df["test2"]
df["test3"]. mean()
cross_corr_rank(df["test3"], df[idy])



df["test1"] = df["D_ask_add_cumsum_30_AP1"] + 1
df["test2"] = df["D_ask_add_cumsum_30_abvAP1"] + 2
df["test3"] = df["test1"] / df["test2"]

df["test1_1"] = df["D_ask_add_cumsum_5_AP1"] + 1
df["test2_1"] = df["D_ask_add_cumsum_5_abvAP1"] + 2
df["test3_1"] = df["test1_1"] / df["test2_1"]
df["test4"] = df["test3"] - df["test3_1"]

cross_corr_rank(df["test4"], df[idy])


a1 = df["D_bid_exch_old_cumsum_30_abvAP1"]
a2 = df["D_ask_exch_old_cumsum_30_abvAP1"]
a3 = (a1 + 1) / (a1 + a2 + 2)


a1 = df["D_bid_exch_old_cumsum_30_blwBP1"]
a2 = df["D_ask_exch_old_cumsum_30_blwBP1"]
a3_1 = (a1 + 1) / (a1 + a2 + 2)

a1 = df["D_bid_exch_old_cumsum_30"]
a2 = df["D_ask_exch_old_cumsum_30"]
a3_1 = (a1 + 1) / (a1 + a2 + 2)

df["test"] = a3 + a3_1
df["test"] = a1 / (a1 + a2)
df["test"] = a3
cross_corr_rank(df["test"], df[idy])



        self.mater("D_ask_add Map(Int64,Int64)", "mapFilValuePos(D_ask_change)")
        self.mater("D_ask_cancel Map(Int64,Int64)", "mapValueNeg(mapFilValueNeg(D_ask_change))")
        self.mater("D_ask_exch Map(Int64,Int64)", "clipMapRight(D_exch,BP1_last)")
        self.mater("D_ask_old Map(Int64,Int64)", "mapSubtract(D_exch,mapFilValuePos(mapSubtract(D_exch,D_ask_last)))")
        self.mater("D_ask_new Map(Int64,Int64)", "mapSubtract(D_ask_exch,D_ask_old)")
        self.mater("D_bid_add Map(Int64,Int64)", "mapFilValuePos(D_bid_change)")
        self.mater("D_bid_cancel Map(Int64,Int64)", "mapValueNeg(mapFilValueNeg(D_bid_change))")
        self.mater("D_bid_exch Map(Int64,Int64)", "clipMapLeft(D_exch,AP1_last)")        
        self.mater("D_bid_old Map(Int64,Int64)", "mapSubtract(D_exch,mapFilValuePos(mapSubtract(D_exch,D_bid_last)))")
        self.mater("D_bid_new Map(Int64,Int64)", "mapSubtract(D_bid_exch,D_bid_old)")
        self.mater("D_mid_exch Map(Int64,Int64)", "clip_map(D_exch,BP1_last+1,AP1_last-1)")

gg = read_sql("select D_bid_old,D_bid_new,D_exch,D_ask_exch,D_bid_exch,AP1_last,BP1_last from rb.tickdata limit 1000")



df["test1"] = df["D_ask_exch_old_cumsum_30_abvAP1"]
df["test2"] = df["D_bid_exch_old_cumsum_30_blwBP1"]
df["test3"] = (df["test1"] + 1) / (df["test1"] + df["test2"] + 2)
df[["test3"] + idy]. corr("spearman")










