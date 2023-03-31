import sys
sys.path.append("/home/lishiyu/Project/bin_tools")
import bins
import pandas as pd
from jump_span import *
from sql import rsq
from append_df import cc2

table_name = "rb.detail"
sql = f"""select
TradingDay,
ExchTimeOffsetUs as time,
Session,
tick_open as open,
WAP as close,
tick_high as high,
tick_low as low,
Volume_DiffLen1 as volume,
Turnover_DiffLen1S as amt,
MX_exch_detail,
MX_ask_exch_old,
MX_ask_exch_new,
MX_bid_exch_old,
MX_bid_exch_new,
MX_moment_exch,
MX_ask_add,
MX_ask_cancel,
MX_bid_add,
MX_bid_cancel,
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
from {table_name}
order by TradingDay, ExchTimeOffsetUs
"""
df = rsq(sql)



def append_TXPV(df):
    res = dict()
    for i in ["exch_detail", "ask_exch_old", "bid_exch_old", "moment_exch",
              "ask_add", "ask_cancel", "bid_add", "bid_cancel"
              ]:
        print(i)
        tmps = df[i]. apply(eval).values
        v, a =dict_sum(tmps)
        res[f"TXPV_a_{i}"] = a
        res[f"TXPV_v_{i}"] = v

        a_cumsum = a.cumsum()
        v_cumsum = v.cumsum()
        a2_cumsum = np.nan_to_num((a ** 2) / v).cumsum()

        for j in [1, 3, 5, 10, 20, 30, 60, 120, 240, 360, 600, 1200, 2400, 3600]:
            k_cnt = np.full_like(v, j)
            np.put(k_cnt, range(j), range(1, j + 1))

            k0 = v_cumsum
            k0_shift = np.roll(k0, j)
            np.put(k0_shift, range(j), 0)
            k0_diff = k0 - k0_shift

            k1 = a_cumsum
            k1_shift = np.roll(k1, j)
            np.put(k1_shift, range(j), 0)
            k1_diff = k1 - k1_shift

            k2 = a2_cumsum
            k2_shift = np.roll(k2, j)
            np.put(k2_shift, range(j), 0)
            k2_diff = k2 - k2_shift

            mean = (k1_diff / k0_diff)        
            sqr = ((k2_diff / k0_diff) - mean**2)

            res[f"TXPV_{i}_pmean_{j}"] = mean
            res[f"TXPV_{i}_psqr_{j}"] = sqr
            res[f"TXPV_{i}_vmean_{j}"] = k0_diff / k_cnt
    return pd.DataFrame(res, index=df.index)


def append_TXPD(df):
    res = dict()
    for j in [1, 3, 5, 10, 20, 30, 60, 120, 240, 360, 600, 1200, 2400, 3600]:
        res[f"TXPD_askbid_pdiff_{j}"] = (df[f"TXPV_ask_exch_old_pmean_{j}"] - df[f"TXPV_bid_exch_old_pmean_{j}"])
        res[f"TXPD_askbid_vdiff_{j}"] = (df[f"TXPV_ask_exch_old_vmean_{j}"] - df[f"TXPV_bid_exch_old_vmean_{j}"])
        res[f"TXPD_askbid_vdiffPorp_{j}"] = res[f"TXPD_askbid_vdiff_{j}"] / df[f"TXPV_exch_detail_vmean_{j}"]
        res[f"TXPD_moment_pdiff_{j}"] = (df[f"TXPV_moment_exch_pmean_{j}"] - df[f"TXPV_exch_detail_pmean_{j}"])
        res[f"TXPD_moment_vPorp_{j}"] = (df[f"TXPV_moment_exch_vmean_{j}"] / df[f"TXPV_exch_detail_vmean_{j}"])

        res[f"TXPD_bid_addcancel_pdiff_{j}"] = df[f"TXPV_bid_add_pmean_{j}"] - df[f"TXPV_bid_cancel_pmean_{j}"]
        res[f"TXPD_bid_addcancel_vdiff_{j}"] = df[f"TXPV_bid_add_vmean_{j}"] - df[f"TXPV_bid_cancel_vmean_{j}"]    
        res[f"TXPD_bid_addcancel_vdiffPorp_{j}"] = res[f"TXPD_bid_addcancel_vdiff_{j}"] / (df[f"TXPV_bid_add_vmean_{j}"] + df[f"TXPV_bid_cancel_vmean_{j}"])

        res[f"TXPD_ask_addcancel_pdiff_{j}"] = df[f"TXPV_ask_add_pmean_{j}"] - df[f"TXPV_ask_cancel_pmean_{j}"]
        res[f"TXPD_ask_addcancel_vdiff_{j}"] = df[f"TXPV_ask_add_vmean_{j}"] - df[f"TXPV_ask_cancel_vmean_{j}"]    
        res[f"TXPD_ask_addcancel_vdiffPorp_{j}"] = res[f"TXPD_ask_addcancel_vdiff_{j}"] / (df[f"TXPV_ask_add_vmean_{j}"] + df[f"TXPV_ask_cancel_vmean_{j}"])
    return pd.DataFrame(res)

from timeseries_detail import append_crossday_return

df = cc2(df, append_TXPV)
df = cc2(df, append_TXPD)
df = cc2(df, append_crossday_return)

df.to_pickle("avg_price.pkl")

################### check point 1 ######################
#df = pd.read_pickle("avg_price.pkl")
#import corr_analyze
#from importlib import reload
#reload(corr_analyze)
from corr_analyze import xydata, beautify_excel
import math

df = xydata(df, x_symbol="^TX")
df.cross_corr()
df.daywise_corr()

import re

def sort_index(tmp_df):
    tmp_index = pd.Series(tmp_df.index, index=tmp_df.index)
    has_param = (~tmp_index.apply(lambda x:re.search("_\d+$", x)).isnull()).values
    pref = tmp_index.copy()        
    surf = pd.Series(0, index=tmp_df.index)
    pref.loc[has_param] = pref.loc[has_param]. apply(lambda x:"_". join(x.split("_")[: -1]))
    surf.loc[has_param] = tmp_index.loc[has_param]. apply(lambda x:int(x.split("_")[ - 1]))
    tmp_df = tmp_df.copy()
    tmp_df["pref"] = pref
    tmp_df["surf"] = surf
    tmp_df.sort_values(["pref", "surf"], inplace=True)
    del tmp_df["pref"]
    del tmp_df["surf"]    
    return tmp_df

with pd.ExcelWriter(f"corr_exch.xlsx", engine='xlsxwriter') as writer:
    tmp_df = df.corrxy.T
    tmp_df = sort_index(tmp_df)
    beautify_excel(tmp_df, "corr", writer)




    for tmp_df, name in zip([df.dcorrxy_sharp.T, df.dcorrxy_mean.T, df.dcorrxy_std.T],
                            ["dcorr_shape", "dcorr_mean", "dcorr_std"]):
        print(name)
        tmp_df = sort_index(tmp_df)        
        v = tmp_df.melt()["value"]. abs()
        maximum = v[v < math.inf]. quantile(0.95)
        beautify_excel(tmp_df, name, writer,
                       conditional_format={
                           'type': '3_color_scale',
                           "min_type": "num",
                           "max_type": "num",
                           "mid_type": "num",                                          
                           "min_value": f"{-maximum}", 
                           "mid_value": "0", 
                           "max_value": f"{maximum}", 
                           "min_color": "red", 
                           "mid_color": "white", 
                           "max_color": "green", 
                       },
                       text_format = {
                           'num_format': '0.00',
                           "font": "Courier New",
                           "font_size": 10,                         
                       })
