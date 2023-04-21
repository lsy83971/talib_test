import numpy as np
import sys
sys.path.append("/home/lishiyu/talib_test/data_process/")
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",2000)
import warnings
from cython_func.split_calc import split_idx, split_cumsum
from IPython.display import display
warnings.filterwarnings("ignore")

from common.show_pv_tab import show_pv,show3
from data_process.local_sql import read_sql,exch_detail
from data_process.kline import table_kline_period

# tb=exch_detail("rb")
# tb.get_columns()
# tb1=table_kline_period("rb",1)
# tb1.get_columns()
def to_edge(df, i, j):
    return (df["to_begin"] > i) & (df["to_end"] > j)

tab1 = exch_detail("rb")
tab1.get_columns()
tab1.col_type["name"]

mpd = {
    "ev6":  "{tab}.D_exch[{tab}.BP1_last+3]", 
    "ev5":  "{tab}.D_exch[{tab}.BP1_last+2]", 
    "ev4":  "{tab}.D_exch[{tab}.BP1_last+1]", 
    "ev3":  "{tab}.D_exch[{tab}.BP1_last]", 
    "ev2":  "{tab}.D_exch[{tab}.BP1_last-1]", 
    "ev1":  "{tab}.D_exch[{tab}.BP1_last-2]",
    
    "v6":  "{tab}.D_ask[{tab}.BP1_last+3]", 
    "v5":  "{tab}.D_ask[{tab}.BP1_last+2]", 
    "v4":  "{tab}.D_ask[{tab}.BP1_last+1]", 
    "v3":  "{tab}.D_bid[{tab}.BP1_last]", 
    "v2":  "{tab}.D_bid[{tab}.BP1_last-1]", 
    "v1":  "{tab}.D_bid[{tab}.BP1_last-2]",
    
    "lv6":  "{tab}.D_ask_last[{tab}.BP1_last+3]", 
    "lv5":  "{tab}.D_ask_last[{tab}.BP1_last+2]", 
    "lv4":  "{tab}.D_ask_last[{tab}.BP1_last+1]", 
    "lv3":  "{tab}.D_bid_last[{tab}.BP1_last]", 
    "lv2":  "{tab}.D_bid_last[{tab}.BP1_last-1]", 
    "lv1":  "{tab}.D_bid_last[{tab}.BP1_last-2]",
}

for i in range(1, 7):
    mpd[f"av{i}"] = f"""{mpd[f"v{i}"]}+{mpd[f"ev{i}"]}-{mpd[f"lv{i}"]}"""



sql = f"""
create table rb.tickdata_wave
engine=MergeTree
order by (date,time)
PARTITION by date
as
-- select
-- e.*,
-- any(e.BP1_last) over (PARTITION by e.wave_symbol order by e.date,e.time) as BP1_wave,
-- any(e.AP1_last) over (PARTITION by e.wave_symbol order by e.date,e.time) as AP1_wave
-- from (


  select
  {" ".join([f"c.{i}," for i in tab1.col_type["name"].tolist()])}
  {(" ".join([f"{j} as {i}," for i,j in mpd.items()])).format(tab='c')}
  b.wave_symbol from
  rb.tickdata as c inner join
  (select
  date,
  time,
  sum(a.wave_symbol) over w1 as wave_symbol
  from
  (
    select date,time,
    case when AP1-BP1=1 and AP1-AP1_last=0 and BP1-BP1_last=0
    and (any(Session) over w0 = Session) then 0
    else 1 end as wave_symbol
    from rb.tickdata as d
    window w0 as (order by d.date,d.time asc Rows BETWEEN 1 PRECEDING and 1 PRECEDING)
  ) as a
  window w1 as (order by a.date,a.time asc Rows BETWEEN UNBOUNDED PRECEDING and 1 PRECEDING) )as b
  on c.date=b.date
  and c.time=b.time

-- ) as e
"""

client.execute(sql)


df = read_sql("""
select
date,time,Session,Symbol,wave_symbol,
v1,
v2,
v3,
v4,
v5,
v6,
ev1,
ev2,
ev3,
ev4,
ev5,
ev6,
lv1,
lv2,
lv3,
lv4,
lv5,
lv6,
av1,
av2,
av3,
av4,
av5,
av6,
AP1,
BP1,
AP1_last,
BP1_last,
MID,
D_ask,
D_bid,
D_exch,
amt,
vol
from rb.tickdata_wave
order by date,time
""")

start_idx = (df["wave_symbol"]. shift(1) != df["wave_symbol"]).astype(np.int8)
res = dict()
res["to_begin"], res["to_end"], res["begin_idx"], res["end_idx"]=split_idx(start_idx.values)
res_df = pd.concat({i: pd.Series(j) for i, j in res.items()}, axis=1)
df = pd.concat([df, res_df], axis=1)
df["next_ret"] = df["MID"]. loc[df["end_idx"]]. values - df["MID"]
v = df["begin_idx"] - 2
v[0] = 0
v[1] = 0
df["last_ret"] = df["MID"] - df["MID"]. loc[v]. values

df["ret60"] = df["MID"]. shift( -60) - df["MID"]
df["ret10"] = df["MID"]. shift( -10) - df["MID"]
df["ret5"] = df["MID"]. shift( -5) - df["MID"]
df["ret1"] = df["MID"]. shift( -1) - df["MID"]

df.to_pickle("/home/lishiyu/talib_test/factor_test/tdf.pkl")
global sbbb
"sbbb"in globals()
sbbb = 1
