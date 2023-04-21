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

from common.show_pv_tab import show_pv,show3
from data_process.local_sql import read_sql,exch_detail
from data_process.kline import table_kline_period

# tb=exch_detail("rb")
# tb.get_columns()
# tb1=table_kline_period("rb",1)
# tb1.get_columns()
def to_edge(df, i, j):
    return (df["to_begin"] > i) & (df["to_end"] > j)

def b3(self,cond,x,y,**kwargs):
    if isinstance(y,str):
        y=[y]
    xy=[x]+y
    tmp=self[xy].loc[cond]
    return tmp.b2(x,y,**kwargs)
pd.DataFrame.b3=b3

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

#client.execute(sql)


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


df["test"] = split_cumsum(d.astype(np.float64).values, start_idx.values)
df["v3_cumsum"] = split_cumsum(df["v3"]. astype(np.float64). values, start_idx.values)
df["ev3_cumsum"] = split_cumsum(df["ev3"]. astype(np.float64). values, start_idx.values)
df["av3_cumsum"] = split_cumsum(df["av3"]. astype(np.float64). values, start_idx.values)

df["v4_cumsum"] = split_cumsum(df["v4"]. astype(np.float64). values, start_idx.values)
df["ev4_cumsum"] = split_cumsum(df["ev4"]. astype(np.float64). values, start_idx.values)
df["av4_cumsum"] = split_cumsum(df["av4"]. astype(np.float64). values, start_idx.values)

df["test"] = split_cumsum(df["ev3"], start_idx.values)

df["v3"]n

c1 = (df["ev3"] > 100)
c2 = (df["av3"]. shift( -1) > df["ev3"])
c1 = (df["ev3"]. shift( -1) < 50)
d = (c1 & c2) & to_edge(df, -1, 1)

df["next_ret"][d]. mean()
df["ret10"][d]. mean()

df["test"] = df["ev4_cumsum"] / df["v4_cumsum"]
df["test1"] = df["ev3_cumsum"] / df["v3_cumsum"]
df["test2"] = df["test"] - df["test1"]
df["test3"] = df["v3"] - df["ev3"] * 3
df["test4"] = df["v2"] / df["v2"]. loc[df["begin_idx"]]. values
df["test5"] = df["av4"] - df["av3"]

df["test6"] = df["v3"]. loc[df["begin_idx"]]. values
df["test7"] = df["v4"]. loc[df["begin_idx"]]. values
df["test8"] = df["test7"] - df["test6"]
df["test9"] = df["ev3"]. shift(1)
df["test10"] = df["ev3"] / (df["ev3_cumsum"] / (df["to_begin"] + 1))

df["test11"] = df["test10"] / 3 + df["ev3"]
df["test12"] = df["ev3"] - df["v3"]

df[df["to_begin"] > 5].b2(x="test", y="ret10", quant=20)
#df[df["to_begin"] > 2].b2(x="test", y="ret10", quant=20)
df[df["to_end"] > 0].b2(x="test", y="ret10", quant=50)
df[df["to_end"] > 0].b2(x="test1", y="ret10", quant=50)
df[df["to_end"] > 0].b2(x="test2", y="ret10", quant=50)
df[df["to_end"] > 0].b2(x="v3", y="ret10", quant=50)

df.b2(x="ev3_cumsum", y="ret10", quant=50)
df[df["to_end"] > 0].b2(x="v3_cumsum", y="ret10", quant=50)
df[df["to_end"] > 0].b2(x="test4", y="ret10", quant=50)
df[df["to_end"] > 0].b2(x="av2", y="ret10", quant=50)

df[["test4", "ret10", "test2", "v3", "v4", "test1", "test3", "ev3", "ev4"]][df["to_end"] > 0]. corr()

df[["ret10", "v3", "v4", "test5", "test6", "test7", "test8"]][df["to_end"] > 0]. corr()
df[["ret10", "ev3_cumsum"]][df["to_end"] > 0]. corr()
df[["ret10", "ev4_cumsum"]][df["to_end"] > 0]. corr()
df[["ret10", "ev4"]][df["to_end"] > 0]. corr()
df[["ret10", "test9"]][df["to_end"] > 0]. corr()
df[["ret10", "test10", "test11", "ev3", "test12"]][df["to_end"] > 0]. corr()


df[df["to_end"] > 0].b2(x="test10", y="ret10", quant=50)
df[(df["to_end"] > 1) & c2].b2(x="test10", y="ret10", quant=50)
df[(df["to_end"] > 1) & c1].b2(x="test10", y="ret10", quant=50)
df[(df["to_end"] > 1) & c1 & c2].b2(x="test10", y="ret10", quant=50)



c1 = (df["av3"] > 300) & (df["av3"] > df["ev3"]. shift(1))
c2 = df["ev3"]. shift(1) > 100
df["ret10"][c1 & to_edge(df, -1, 0)]. mean()
df["ret10"][c1 & to_edge(df, -1, 0)]. mean()
df["ret10"][c2 & c1 & to_edge(df, -1, 0)]. mean()

c1 = df["ev3_cumsum"] / (df["to_begin"] + 1) < 20
c2 = df["ev4_cumsum"] / (df["to_begin"] + 1) < 20
c3 = df["ev3"]. shift( -1) > 100

df1 = df[c2 & c1 & c3 & to_edge(df, -1, 1)]


df1.b2("", "ret10")

c1 = df["av4_cumsum"] - df["ev4_cumsum"] > -10
c2 = df["av4_cumsum"] > 1000
c3 = df["av3_cumsum"] < 300


df["ret10"][c1 & c2 & to_edge(df, 0, 0)]. mean()
df["ret10"][c1 & c2 & c3 & to_edge(df, 0, 0)]. mean()
df["ret10"][c2 & c3 & to_edge(df, 0, 0)]. mean()
df["ret10"][c2 & c3 & to_edge(df, 0, 0)]. count()


c1 = (df["av4"]. shift(1) < 10) & (df["ev4"]. shift(1) > 100)
c2 = df["av4"] > df["ev4"]. shift(1)


df["ret10"][c1 & c2 & to_edge(df, 0, 0)]. mean()
df.index[c1 & c2 & to_edge(df, 0, 0)]

show3(df, 2550, 30)

c1 = df["ev3"]. shift(1) > 50
c2 = df["av3"] < 0
c3 = df["v3"]. shift(2) > 300



df["ret10"][c1 & to_edge(df, 0, 0)]. mean()
df["ret10"][c2 & to_edge(df, 0, 0)]. mean()
df["ret10"][c1 & c2 & to_edge(df, 0, 0)]. mean()

df[to_edge(df, 0, 0)]. b2("av3", "ret10")



c1 = df["ev4"]. shift(1) > 50
c2 = (df["av4"]. shift(1) < 10) & (df["av4"] > 70)
c3 = df["av4"] > (df["ev4"] + df["ev4"]. shift(1) + 20)
c4 = df["av4"] > (df["ev4"])

df["ret10"][c1 & c2 & c3 & to_edge(df, 0, 0)]. mean()
df["ret10"][c2 & c3 & to_edge(df, 0, 0)]. mean()


df["ret10"][c3 & to_edge(df, 0, 0)]. mean()
df["ret10"][c4 & to_edge(df, 0, 0)]. mean()
df["ret10"][c3 & to_edge(df, 0, 0)]. count()
df["ret10"][c4 & to_edge(df, 0, 0)]. count()

df["test"] = (df["ev4_cumsum"] / df["v4_cumsum"]). fillna( -1)
df["test1"] = df["ev4"] / df["v4"] - (df["ev4_cumsum"] / df["v4_cumsum"])
df["test2"] = df["ev4"] / df["v4"]
df["test3"] = df["ev4"] - df["ev3"]
df[to_edge(df, 0, 0)].b2("test", "ret10", quant=20)
df[to_edge(df, 0, 0)].b2("test1", "ret10", quant=20)
df[to_edge(df, 0, 0)].b2("test2", "ret10", quant=20)
df[to_edge(df, 0, 0)].b2("test3", "ret10", quant=20)

c1 = df["test3"] > 50
df[c1 & to_edge(df, 0, 0)].b2("ev3", "ret10", quant=20)
df[to_edge(df, 0, 0)].b2("test", "ret10", quant=100)
df[to_edge(df, 0, 0)][["test", "ret10"]]. corr()
df[to_edge(df, 0, 0)][["ret10", "test", "test1", "test2", "test3"]]. corr()

df["test4"] = df["ev4"]. shift(1)
df["test5"] = df["ev4"]. shift(2)
df["test6"] = df["ev4"]. shift(3)
df["test7"] = df["ev4"]. shift(4)
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=20)
df[to_edge(df, 1, 0)].b2("test5", "ret10", quant=20)
df[to_edge(df, 2, 0)].b2("test6", "ret10", quant=20)
df[to_edge(df, 3, 0)].b2("test7", "ret10", quant=20)

df["test4"] = df["av4"]. shift(1)
df["test5"] = df["av4"]. shift(2)
df["test6"] = df["av4"]. shift(3)
df["test7"] = df["av4"]. shift(4)

df["test4"] = df["v4"]. shift(1) - df["v3"]. shift(1)
df["test5"] = df["v4"]. shift(2) - df["v3"]. shift(2)
df["test6"] = df["v4"]. shift(3) - df["v3"]. shift(3)
df["test7"] = df["v4"]. shift(4) - df["v3"]. shift(4)

df[to_edge(df, 3, 0)][["ret10", "test4", "test5", "test6", "test7"]]. corr()

df["test4"] = df["v4"]. shift(1) - df["v3"]. shift(1)
df["test5"] = df["v4"]. shift(2) - df["v3"]. shift(2)
df["test6"] = df["v4"]. shift(3) - df["v3"]. shift(3)
df["test7"] = df["v4"]. shift(4) - df["v3"]. shift(4)


df[to_edge(df, -1, 0)].b2("v5", "ret10", quant=20)
df[to_edge(df, -1, 0)].b2("av5", "ret10", quant=50)

idx_last = (df["begin_idx"] - 2)
idx_last[0] = 0
idx_last[1] = 0
df["last_dif"] = df.loc[df["begin_idx"], "MID"]. values - df.loc[idx_last, "MID"]. values

df[to_edge(df, 0, 0)].b2("last_dif", "ret10", quant=50)
df["last_dif"]


df["up1"] = df["ev4"] + df["av3"]
df["down1"] = df["ev3"] + df["av4"]

v = (df["up1"]. shift(1)* df["down1"]).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df["up1"] * df["up1"]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df["down1"] * df["down1"]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=50)


v = (df["up1"]. shift(1)* df["down1"]).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df["ev4"] * df["ev4"]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df["av4"] * df["av4"]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=50)

v = (df["up1"]* df["down1"]. shift(1)).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df["ev4"] * df["ev4"]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df["av4"] * df["av4"]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=50)

v = (df["ev4"]. shift(1)* df["av4"]).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df["ev4"] * df["ev4"]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df["av4"] * df["av4"]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df[to_edge(df, 1, 0)].b2("test4", "ret10", quant=50)


v = (df["ev4"]* df["av4"]. shift(1)).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df["ev4"] * df["ev4"]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df["av4"] * df["av4"]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=50)


id1 = "ev4"
id2 = "v4"

id1 = "v4"
id2 = "v4"


id1 = "av4"
id2 = "ev4"


v = (df[id1]* df[id2]. shift(1)).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df[id1] * df[id1]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df[id2] * df[id2]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=100)

i = df.index[df["test4"] > 0.7][10001]
print(i)
print(df["ret10"][i])
show3(df, i + 5, 20)


df["test4"] = df["ev4"] / (df["ev4_cumsum"] + 100) * (1 + df["to_begin"])
df[to_edge(df, 0, 0)].b2("test4", "ret10", quant=100)
df[to_edge(df, 0, 0)].b2("ev4", "ret10", quant=100)
df[to_edge(df, 0, 0)].b2("v4", "ret10", quant=100)


df[["v4", "ev4"]]. corr()




df[to_edge(df, 0, 0)].b2("ev5", "ret10", quant=100)

id1 = "av4"
id2 = "v4"

v = (df[id1]* df[id2]. shift(1)).astype(np.float64)
v.loc[start_idx == 1] = 0
df["test1"] = split_cumsum(v.values, start_idx.values)
df["test2"] = split_cumsum((df[id1] * df[id1]).astype(np.float64).values, start_idx.values)
df["test3"] = split_cumsum((df[id2] * df[id2]).astype(np.float64).values, start_idx.values)
df["test4"] = (df["test1"] ** 2) / (df["test2"] * df["test3"])
df["test5"] = df["test1"] / df["test2"]
df["test6"] = df["test1"] / df["test3"]
df["test"] = df["ev4"] / df["v4"]. shift(1)

df[to_edge(df, 0, 0)].b2("ev4", ["ret1", "ret5", "ret10"], quant=50)

## 1.

c1 = (df["av2"] < -20) & (df["av3"] > ( -df["av2"]))
df["ret10"][c1 & to_edge(df, 0, 1)]. mean()
#df["ret10"][c1 & to_edge(df, 0, 0)]. mean()
idy = ["ret1", "ret5", "ret10", "ret60"]
df[c1 & to_edge(df, 0, 0)]. b2(x="to_begin", y=idy)



df[to_edge(df, 0, 0)].b2("ev4", , quant=50)



df[to_edge(df, 0, 0)]. b2(x="ev3", y=idy, quant=20)
df[to_edge(df, 0, 0)]. b2(x="v3", y=idy, quant=20)
df[to_edge(df, 0, 0)]. b2(x="av3", y=idy, quant=20)



from sklearn.linear_model import Lasso

las = Lasso(alpha=0.001)
x = df.loc[to_edge(df, 0, 0), ["ev3", "av3", "v3", "ev4", "av4", "v4"]]
x = (x - x.mean()) / x.std()
y = df.loc[to_edge(df, 0, 0), "ret10"]

x = x.loc[~(y.isnull())]
y = y.loc[~y.isnull()]
las.fit(x, y)
y1 = pd.Series(las.predict(x), index=y.index)

sb = pd.concat([y, y1], axis=1)
sb.b2(0, "ret10", quant=100)

import lightgbm as lgb
from bin_tools.logit import cond_part

x = df.loc[to_edge(df, 0, 0) & (~df["ret10"]. isnull()), ["ev3", "av3", "v3", "ev4", "av4", "v4", "to_begin"]]
y = df.loc[to_edge(df, 0, 0) & (~df["ret10"]. isnull()), "ret10"]

c1, c2, c3=cond_part(pd.Series(y.index), [0.6, 0.8])
x1, y1=x.loc[c1], y.loc[c1]
x2, y2=x.loc[c2], y.loc[c2]
x3, y3=x.loc[c3], y.loc[c3]

dt_train = lgb.Dataset(x1, y1)
dt_valid = lgb.Dataset(x2, y2)
dt_test = lgb.Dataset(x3, y3)

params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 8,
    'learnnig_rage': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    "n_estimators": 50,
    "l2": 10,
    "l1": 10, 
}
est = lgb.train(params,
                train_set=dt_train,
                valid_sets=(dt_train, dt_valid),
                early_stopping_rounds=30)

pd.concat([pd.Series(est.predict(x3), index=y3.index), y3], axis=1).corr()

## 1 

c1 = (df["av3"] > 50) & (df["last_ret"] > 0)
#c1 = (df["ev3"] < 20) & (df["av3"] > 50) & (df["last_ret"] < 0)
df[c1 & to_edge(df, -1, 0)]. b2(x="to_begin", y=idy)
df[to_edge(df, -1, 0)]. b2(x="to_begin", y=idy)


## 2

c1 = (df["ev3"] < 20) & (df["av3"] > 50) & (df["last_ret"] < 0) & (df["to_begin"] <= 4)
c1 = (df["ev3"] < 20) & (df["last_ret"] < 0)
c1 = (df["av3"] > 50) & (df["last_ret"] > 0)
#c1 = (df["last_ret"] > 0)
df[c1 & to_edge(df, -1, 0)]. b2(x="av4", y=idy)
df[c1 & to_edge(df, -1, 0)]. b2(x="v4", y=idy)
df[c1 & to_edge(df, -1, 0)]. b2(x="v3", y=idy)



"{:.2g}". format(0.3)



start_idx = (df["wave_symbol"]. shift(1) != df["wave_symbol"]).astype(np.int8)
df["test"] = split_cummax(df["ev3"].astype(np.float64).values, start_idx.values)
df["test1"] = df["ev3"] / df["test"]. shift(1)
df["test2"] = df["ev3"]. shift(1)
#df[to_edge(df, 10, 0)].b2(x="test", y=idy, quant=20)
df[to_edge(df, 5, 0)].b2(x="test1", y=idy, quant=40)

start_idx = (df["wave_symbol"]. shift(1) != df["wave_symbol"]).astype(np.int8)
df["test"] = split_cummax(df["av3"].astype(np.float64).values, start_idx.values)
df["test1"] = df["av3"] / df["test"]. shift(1)
df["test2"] = df["av3"]. shift(1)
df[to_edge(df, 10, 0)].b2(x="test", y=idy, quant=20)
df[to_edge(df, 5, 0)].b2(x="test1", y=idy, quant=20)


start_idx = (df["wave_symbol"]. shift(1) != df["wave_symbol"]).astype(np.int8)
df["test1"] = split_cummax(df["ev3"].astype(np.float64).values, start_idx.values)
df["test0"] = split_cummax(df["ev4"].astype(np.float64).values, start_idx.values)
df["test1_2"] = df["test1"]. shift(2)
df["test1_3"] = df["test1"]. shift(3)

df["test0_2"] = df["test0"]. shift(2)
df["test2"] = df["ev3"]. shift(1)

df[to_edge(df, 3, 0) & (df["test1_2"] < 20) & (df["test0_2"] < 20) & (df["test2"] >= 40)].b2(x="av3", y=idy, quant=20)
df[to_edge(df, 3, 0) & (df["test1_2"] < 100) & (df["test2"] >= 40)].b2(x="av3", y=idy, quant=20)
df[to_edge(df, 3, 0) & (df["test1_2"] > 100) & (df["test2"] >= 40)].b2(x="av3", y=idy, quant=20)
df[to_edge(df, 3, 0) & (df["test2"] >= 40)].b2(x="av3", y=idy, quant=20)
df[to_edge(df, 3, 0)].b2(x="av3", y=idy, quant=20)


df.b3(to_edge(df, 3, 0), x="test1_2", y=idy, quant=20)
df[to_edge(df, 3, 0)].b2(x="test1_3", y=idy, quant=20)

df[to_edge(df, 3, 0) & (df["test1_2"] > 100)].b2(x="av3", y=idy, quant=20)

df.b3(to_edge(df, 3, 0) & (df["test1_2"] > 100), x="av3", y=idy, quant=20)
df.b3(to_edge(df, 3, 0), x="av3", y=idy, quant=20)

df[to_edge(df, 3, 0) & (df["test1_2"] < 20)& (df["test0_2"] < 20)].b2(x="av3", y=idy, quant=20)

df[(df["last_ret"] > 0) & to_edge(df, 5, 0) & (df["test1"] < 15 )& (df["test0"] < 15) & (df["test2"] > 50)].\
    b2(x="av3", y=idy, quant=20)

df[(df["last_ret"] < 0) & to_edge(df, 5, 0) & (df["test1"] < 20 ) & (df["test2"] > 30)].\
    b2(x="av3", y=idy, quant=20)


a1 = (df["ev4"] + df["ev5"])
a2 = (df["v4"] + df["v5"])


a2 = (df["ev4"] + df["ev5"])
a1 = (df["av4"] + df["av5"])

sb12 = (a1*a2.shift(1)).cumsum()
sb11 = (a1 * a1).cumsum()
sb22 = (a2 * a2).cumsum()

df["test1"] = (sb12 - sb12.shift(10)) / (sb11 - sb11.shift(10))
df["test2"] = (sb12 - sb12.shift(10)) / (sb22 - sb22.shift(10))
df["test3"] = (sb12 - sb12.shift(10)) / (((sb22 - sb22.shift(10))*(sb11 - sb11.shift(10)))**(1 / 2))

df.b2("test3", idy, quant=50)
df.b2("test1", idy)
df.b2("test2", idy)
show_excel(df, i, lag, path, tab="tab", mode='a')
pd.options.display.float_format = '{:.3f}'.format

start_idx = (df["wave_symbol"]. shift(1) != df["wave_symbol"]).astype(np.int8)
df["ev3_cummax"] = split_cummax(df["ev3"].astype(np.float64).values, start_idx.values)
df["ev4_cummax"] = split_cummax(df["ev4"].astype(np.float64).values, start_idx.values)
c1 = ((df["to_begin"] > 5) & (df["ev3_cummax"]. shift(1) < 30) & (df["ev4_cummax"]. shift(1) < 30) & (df["ev3"] > 40))
c1.mean()

pd.options.display.float_format = '{:.3f}'.format
df[idy]. loc[c1 & to_edge(df, 1, 1)]. mean()


gg = c1[c1]. index[400]
pd.options.display.float_format = '{:.0f}'.format
print(gg)
print(show2(df, gg + 90, 100))




