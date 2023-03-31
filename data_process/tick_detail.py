import pandas as pd
import numpy as np
import math
from append_df import cc1, cc2
import sys
sys.path.append("/home/lishiyu/Project/bin_tools")
from bins import binning, bins_simple_mean
from pvdict_utils import *

def append_cols(df):
    #df = cc2(df, append_cols)    
    res = dict()
    res["WAP"] = (df["AskPrice1"] * (df["BidVolume1"] + 1) + \
                  df["BidPrice1"] * (df["AskVolume1"] + 1)) / \
                  (df["BidVolume1"] + df["AskVolume1"] + 2)
    res["WAP_Lag1"] = res["WAP"]. shift(1)
    res["MidPriceLvl1_Lag1"] = df["MidPriceLvl1"]. shift(1)
    
    res["AskBidPrice_Intv"] = (df["AskPrice1"]-df["BidPrice1"]-1)
    res["AskBidPrice_Intv_Lag1"] = res["AskBidPrice_Intv"].shift(1)
    res["BidPrice1_Diff1"] = df["BidPrice1"]-df["BidPrice1_Lag1"]
    res["AskPrice1_Diff1"] = df["AskPrice1"]-df["AskPrice1_Lag1"]
    res["is_stable"] = ((res["BidPrice1_Diff1"] == 0) & \
                        (res["AskPrice1_Diff1"] == 0) & \
                        (res["AskBidPrice_Intv"] == 0))
    
    v1 = (res["is_stable"]).cumsum()
    v2 = v1.copy()
    v2.loc[res["is_stable"]] = None
    v2_2 = v2.fillna(method="bfill")
    v2_1 = v2.fillna(method="ffill")
    v3_1 = (pd.Series(df.index) - (v1 - v2_1)).astype(int)
    v3_2 = (pd.Series(df.index) + (v2_2 - v1) + res["is_stable"]).\
        fillna(v1.shape[0] - 1).astype(int)
    
    res["last_unstable"] = v3_1
    res["F_next_unstable"] = v3_2
    res["index"] = pd.Series(df.index)

    idx = res["index"] - 1
    idx.iloc[0] = 0
    res["last_unstable1"] = res["last_unstable"]. loc[idx]. reset_index(drop=True)

    idx = res["index"] + 1
    idx.iloc[ - 1] = idx.shape[0] - 1
    res["F_next_unstable1"] = res["F_next_unstable"]. loc[idx]. reset_index(drop=True)

    res["tickcnt_fromlast"] = (res["index"] - res["last_unstable1"])
    res["F_tickcnt_tillnext"] = (res["F_next_unstable1"] - res["index"])
    return res

def append_tick_exch(x):
    res = dict()
    res["ask_complete"] = add_dict(x["ask_dict"], x["exch_detail"])
    res["ask_change"] = sub_dict(res["ask_complete"], x["ask_dict_lag1"])
    res["ask_add"] = pos_dict(res["ask_change"])
    res["ask_cancel"] = inverse_dict(neg_dict(res["ask_change"]))
    res["ask_exch_old"] = pos_dict(min_dict(x["exch_detail"], x["ask_dict_lag1"]))
    res["ask_exch_new"] = pos_dict(sub_dict(x["exch_detail"], res["ask_exch_old"]))

    res["bid_complete"] = add_dict(x["bid_dict"], x["exch_detail"])
    res["bid_change"] = sub_dict(res["bid_complete"], x["bid_dict_lag1"])
    res["bid_add"] = pos_dict(res["bid_change"])
    res["bid_cancel"] = inverse_dict(neg_dict(res["bid_change"]))
    res["bid_exch_old"] = pos_dict(min_dict(x["exch_detail"], x["bid_dict_lag1"]))
    res["bid_exch_new"] = pos_dict(sub_dict(x["exch_detail"], res["bid_exch_old"]))
    
    res["moment_exch"] = sub_dict(res["bid_exch_new"], res["ask_exch_old"])
    res["exch_detail"] = x["exch_detail"]

    dict_names = list(res.keys()).copy()
    for i in dict_names:
        if len(res[i]) > 0:
            res[i + "_max"] = max(res[i])
            res[i + "_min"] = min(res[i])
            res[i + "_amt"] = sum_amt(res[i])
            res[i + "_volume"] = sum_volume(res[i])
            if res[i + "_volume"] > 0:
                res[i + "_avgp"] = res[i + "_amt"] / res[i + "_volume"]
        else:
            res[i + "_amt"] = 0
            res[i + "_volume"] = 0

    del res["exch_detail"]
    return res

def append_exch_cumsum_detail(df):
    res = dict()
    for i in ["exch_detail",
              "ask_exch_old", "ask_exch_new",
              "bid_exch_old", "bid_exch_new",
              "moment_exch",
              "ask_add",
              "ask_cancel",
              "bid_add",
              "bid_cancel",
              ]:

        tab = pd.DataFrame(df[i]. tolist()).fillna(0).cumsum()
        tab_diff = (tab - tab.loc[df["last_unstable1"]]. reset_index(drop=True))
        res[i + "_cumsum"] = tab
        res[i + "_cumsum_difflast"] = (tab - tab.loc[df["last_unstable1"]]. reset_index(drop=True))
        res["F_" + i + "_cumsum_diffnext"] = (tab.loc[df["F_next_unstable1"]]. reset_index(drop=True) - tab)
        res[i + "_cumsum_difflast1"] = (tab.shift(1).fillna(0) - tab.loc[df["last_unstable1"]]. \
                                        reset_index(drop=True))
        res["F_" + i + "_cumsum_diffnext1"] = (tab.loc[df["F_next_unstable1"]]. \
                                               reset_index(drop=True).shift(1).fillna(0) - tab)
    return res

def append_exch_cumsum(df):
    exch_cumsum_info = append_exch_cumsum_detail(df)
    res = dict()
    # if not hasattr(df, "attr"):
    #     df.attr = dict()
    for i, j in exch_cumsum_info.items():
        # df.attr[i] = j
        res[i + "_volume"] = j.sum(axis=1)
        res[i + "_amt"] = (j * j.columns).sum(axis=1)
        res[i + "_avgp"] = res[i + "_amt"] / res[i + "_volume"]
        res[i] = pd.Series(j.T.to_dict()).apply(nozero_dict)
    return res
    
def append_dev(df):
    cols = df.cc("avgp").ncc("^F_|Ret").ncc("_dev$")
    res = dict()
    a1 = df["AskPrice1"]
    for i in cols:
        res[i + "_dev"] = df[i] - a1
    return res

def append_nexttick(df):
    #df = cc2(df, append_nexttick)
    res = dict()
    for i in ["exch_detail",
              "ask_exch_old", "ask_exch_new",
              "bid_exch_old", "bid_exch_new",
              "moment_exch",
              "ask_add",
              "ask_cancel",
              "bid_add",
              "bid_cancel",
              ]:
        res["F_" + i + "_nexttick"] = df[i]. shift( -1)
    return res

def append_above_below_price(x):
    #df = cc1(df, append_above_below_price)
    res = dict()
    for i in ["exch_detail",
              "ask_exch_old", "ask_exch_new",
              "bid_exch_old", "bid_exch_new",
              "moment_exch",
              "ask_add",
              "ask_cancel",
              "bid_add",
              "bid_cancel",
              ]:
        for j in [i, "F_" + i + "_cumsum_diffnext", i + "_cumsum_difflast"]:
            res[j + "_aboveA1"] = sum(cut_dict(x[j], l1=x["AskPrice1"]).values())
            res[j + "_belowB1"] = sum(cut_dict(x[j], l2=x["BidPrice1"]).values())
    return res

def append_future_exch(df):
    #df = cc2(df, append_future_exch)
    res = dict()
    res["F_can_dealwith_A1"] = (df["F_exch_detail_cumsum_diffnext_aboveA1"] - df["AskVolume1"]) > 0
    res["F_can_dealwith_B1"] = (df["F_exch_detail_cumsum_diffnext_belowB1"] - df["BidVolume1"]) > 0
    return res

def append_last_next(df):
    #df = cc2(df, append_last_next)
    res = dict()
    ## TODO cumsum can diff
    res["AskPrice1_Difflast"] = (df["AskPrice1"]. loc[df["last_unstable1"]]. reset_index(drop=True) - df["AskPrice1"])
    res["BidPrice1_Difflast"] = (df["BidPrice1"]. loc[df["last_unstable1"]]. reset_index(drop=True) - df["BidPrice1"])
    res["WAP_Difflast"] = df["WAP"]. loc[df["last_unstable1"]]. reset_index(drop=True) - df["WAP"]
    res["VWAP_Difflast"] = df["VWAP"]. loc[df["last_unstable1"]]. reset_index(drop=True) - df["VWAP"]

    res["F_AskPrice1_Diffnext"] = (df["AskPrice1"] - df["AskPrice1"]. loc[df["F_next_unstable1"]]. reset_index(drop=True))
    res["F_BidPrice1_Diffnext"] = (df["BidPrice1"] - df["BidPrice1"]. loc[df["F_next_unstable1"]]. reset_index(drop=True))
    res["F_WAP_Diffnext"] =  df["WAP"] - df["WAP"]. loc[df["F_next_unstable1"]]. reset_index(drop=True)
    res["F_VWAP_Diffnext"] =  df["VWAP"] - df["VWAP"]. loc[df["F_next_unstable1"]]. reset_index(drop=True)    
    return res

def append_tick_detail(df):
    df = cc2(df, append_cols)
    df = cc1(df, append_tick_exch)
    df = cc2(df, append_exch_cumsum)

    df = cc2(df, append_dev)
    df = cc2(df, append_nexttick)
    df = cc1(df, append_above_below_price)
    df = cc2(df, append_future_exch)
    df = cc2(df, append_last_next)
    return df


if __name__ == "__main__":
    df = df0.copy()
    df = append_feature_exch_detail(df)
    df = append_tick_detail(df)

    df.dtypes.value_counts()
    #df.dtypes.value_counts()





