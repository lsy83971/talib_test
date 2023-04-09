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

def cross_corr(dfx, dfy):
    dfx_v = (~dfx.isnull()).astype(np.float64)
    dfy_v = (~dfy.isnull()).astype(np.float64)
    dfx = dfx.fillna(0)
    dfy = dfy.fillna(0)

    corr_crosssum = dfy.T @ dfx
    corr_vcnt = (dfy_v.T) @ (dfx_v)
    corr_xsum = (dfy_v.T) @ (dfx)
    corr_ysum = (dfy.T) @ (dfx_v)

    # if fix_ymean:
    #     corr_ysum[:] = 0

    corr_xmean = (corr_xsum / corr_vcnt)
    corr_ymean = (corr_ysum / corr_vcnt)

    corr_xsqr = dfy_v.T @ (dfx**2)
    corr_ysqr = (dfy.T**2) @ (dfx_v)

    corr_xsqr_N = corr_xsqr - (corr_xmean**2) * (corr_vcnt)
    corr_ysqr_N = corr_ysqr - (corr_ymean**2) * (corr_vcnt)
    corr_crosssum_N = corr_crosssum - (corr_xmean * corr_ymean * corr_vcnt)
    corr_xy = corr_crosssum_N / ((corr_ysqr_N * corr_xsqr_N)**(1 / 2))
    return corr_xy

def daywise_corr(data, d, xidx, yidx):
    ls_d = list()
    ls_data = list()
    for i, x in data.groupby(d):
        ls_d.append(i)
        ls_data.append(cross_corr(x[xidx], x[yidx]))
        
    daywise_corr_df = np.array(ls_data)
    daywise_mean = pd.DataFrame(np.nanmean(daywise_corr_df, axis=0), index=yidx, columns=xidx)
    daywise_std = pd.DataFrame(np.nanstd(daywise_corr_df, axis=0), index=yidx, columns=xidx)
    daywise_sharp = daywise_mean / daywise_std
    return daywise_sharp, daywise_mean, daywise_std, daywise_corr_df

def erase_kline_tickdata(data, idx, idy, split_type=1):
    ### type 0 group by date
    ### type 1 group by (date (morning, afternoon,night))
    ### type 2 group by (date (day, night))
    
    erase_y = {
        "ORT20": 20,
        "ORT40": 40,
        "ORT60": 60,
        "ORM1": 120,
        "ORM3": 360,
        "ORM5": 600,
        "ORM10": 1200,
        "ORM15": 1800,
        "ORM20": 2400,
        "ORM30": 3600,
        "ORM40": 4800,
        "ORM50": 6000,
        "ORM60": 7200,
    }

    erase_x = {
        "TX_sma_10": 10,
        "TX_sma_30": 30,
        "TX_sma_60": 60,
        "TX_sma_120": 120,
        "TX_sma_180": 180,
        "TX_sma_240": 240,
        "TX_sma_360": 360,
        "TX_sma_480": 480,
        "TX_sma_600": 600,
        "TX_sma_720": 720,
        "TX_sma_960": 960,
        "TX_sma_1200": 1200,
        "TX_sma_1800": 1800,
        "TX_sma_2400": 2400,
        "TX_sma_3600": 3600,
        "TX_ema_10": 60,
        "TX_ema_30": 60,
        "TX_ema_60": 120,
        "TX_ema_120": 240,
        "TX_ema_180": 360,
        "TX_ema_240": 480,
        "TX_ema_360": 720,
        "TX_ema_480": 960,
        "TX_ema_600": 600,
        "TX_ema_720": 720,
        "TX_ema_960": 960,
        "TX_ema_1200": 1200,
        "TX_ema_1800": 1800,
        "TX_ema_2400": 2400,
        "TX_ema_3600": 3600,
    }

    for i in idx:
        erase_period = erase_x.get(i, 60)
        data.loc[data[f"from_begin{split_type}"] <= erase_period / 2, i] = None

    for i in idy:
        erase_period = erase_y.get(i, 60)
        data.loc[data[f"to_end{split_type}"] <= erase_period / 2, i] = None

def beautify_excel(tmp_df,
                   sheet_name, 
                   writer, conditional_format=None, text_format=None, header_format=None):
    tmp_df.to_excel(writer, sheet_name=sheet_name)
    if conditional_format is None:
        conditional_format = {
            'type': '3_color_scale',
            "min_type": "num",
            "max_type": "num",
            "mid_type": "num",                                          
            "min_value": "-0.1", 
            "mid_value": "0", 
            "max_value": "0.1", 
            "min_color": "red", 
            "mid_color": "white", 
            "max_color": "green", 
    }
    if text_format is None:
        text_format = {
            'num_format': '0.00%',
            "font": "Courier New",
            "font_size": 10,                         
        }
    if header_format is None:
        header_format = {
            "font": "Courier New",
            "font_size": 10, 
        }
    worksheet = writer.book.get_worksheet_by_name(sheet_name)
    worksheet.conditional_format(0, 0, tmp_df.shape[0], tmp_df.shape[1], conditional_format)
    text_format = writer.book.add_format(text_format)
    worksheet.set_column(1, tmp_df.shape[1], None, text_format)
    header_format = writer.book.add_format(
        {
            "font": "Courier New",
            "font_size": 10, 
        }
    )
    for i in range(tmp_df.shape[1]):
        worksheet.write(0, i + 1, str(tmp_df.columns[i]), header_format)

    for i in range(tmp_df.shape[0]):
        worksheet.write(i + 1, 0, str(tmp_df.index[i]), header_format)        


def get_idx_info(x:str):
    l = list()
    l1 = x.split("_")
    for j, i in enumerate(l1[:: -1]):
        try:
            l.append(float(i))
        except:
            break_point = j
            break
    return tuple(["_". join(l1[: -break_point])] + l[:: -1])

def sort_index(tmp_df):
    return tmp_df.loc[pd.Series({i:get_idx_info(i) for i in tmp_df.index}).sort_values().index]


        
class xydata(pd.DataFrame):
    def __init__(self, data, x_symbol="^TX", y_symbol="^ORM|^ORT"):
        super().__init__(data)
        self.idx = self.cc(x_symbol)
        self.idy = self.cc(y_symbol)
        self.date = data["TradingDay"]. drop_duplicates().sort_values()
    
    def erase_kline_tick_data(self, split_type=1):
        ### type 0 group by date
        ### type 1 group by (date (morning, afternoon,night))
        ### type 2 group by (date (day, night))
        erase_kline_tickdata(self, self.idx, self.idy, split_type=split_type)

    def cross_corr(self):
        self.corrxy = cross_corr(self[self.idx], self[self.idy])

    def daywise_corr(self, d="TradingDay"):
        self.dcorrxy_sharp, self.dcorrxy_mean, self.dcorrxy_std, self.dcorrxy_corr = \
            daywise_corr(self, d, self.idx, self.idy)

    def to_excel(self, path, append_info=None):
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            tmp_df = self.corrxy.T
            tmp_df = sort_index(tmp_df)
            beautify_excel(tmp_df, "corr", writer)
            for tmp_df, name in zip([self.dcorrxy_sharp.T,
                                     self.dcorrxy_mean.T,
                                     self.dcorrxy_std.T],
                                    ["dcorr_sharpe", "dcorr_mean", "dcorr_std"]):
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
                
            if append_info is not None:
                for i, j in append_info.items():
                    beautify_excel(j, i, writer,
                                   conditional_format={},
                                   text_format = {
                                       "font": "Courier New",
                                       "font_size": 10,                         
                                   })
                    

if __name__ == "__main__":
    from timeseries_detail import func_info
    data = get_kline("rb.detail")
    data = cc2(data, append_crossday_return)
    Ry = data.cc("RT|RM").ncc("ORT|ORM").tolist()
    data = cc2(data, append_techIdxBasic)
    data = cc2(data, append_MAIdx)

    df = xydata(data)
    df.erase_kline_tick_data()
    df.cross_corr()
    df.daywise_corr()


    #for period in [1, 5, 10, 30, 60]:
    for period in [1, 5, 10, 30, 60]:        
        data = get_kline("rb.detail", period)
        data = cc2(data, append_crossday_return)
        data = cc2(data, append_techIdxBasic)
        data = cc2(data, append_MAIdx)

        df = xydata(data)
        df.cross_corr()
        df.daywise_corr()


        with pd.ExcelWriter(f"corr_period_{period}.xlsx", engine='xlsxwriter') as writer:
            tmp_df = df.corrxy.T
            beautify_excel(tmp_df, "corr", writer)
            
            for tmp_df, name in zip([df.dcorrxy_sharp.T, df.dcorrxy_mean.T, df.dcorrxy_std.T],
                                    ["dcorr_sharpe", "dcorr_mean", "dcorr_std"]):
                print(name)
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
            beautify_excel(func_info, "function_info", writer,
                           conditional_format={},
                           text_format = {
                               "font": "Courier New",
                               "font_size": 10,                         
                           })


    

