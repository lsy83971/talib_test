import pickle
import datetime
import os
import requests
import scipy
import math
import time
import multiprocessing
import itertools
import json
import warnings
import glob

import numpy as np
import pandas as pd

import sys
sys.path.append("./tick_strategy")
from data_utils import *
from feature import *

manager = multiprocessing.Manager()

def get_ts_info(start_time,end_time,symbol_class):
    trading_time_interval_dict = {
        'night': (3600 * -3, 3600 * -1),
        'day1': (3600 * 9, 3600 * 10 + 60 * 15),
        'day2': (3600 * 10 + 60 * 30, 3600 * 11 + 60 * 30),
        'day3': (3600 * 13 + 60 * 30, 3600 * 15)
    }
    tick_size_in_yuan = 1    
    volume_multiple = 10
    exchange = 'SHFE'    
    user = ''
    password = ''
    is_new=False

    start_time=pd.to_datetime(start_time)
    end_time=pd.to_datetime(end_time).strftime('%Y-%m-%d')
    global manager
    shared_raw_data_dict = manager.dict()
    process_list = list()
    i = 0
    while True:
        date = (start_time + datetime.timedelta(i)).strftime('%Y-%m-%d')
        i += 1
        if date > end_time:
            break
        #max_symbol = get_max_symbol_new(date, symbol_class, exchange, user, password)
        max_symbol = get_max_symbol(date, symbol_class)        
        print(date, max_symbol)
        process = multiprocessing.Process(target=append_data,
                                          args=(shared_raw_data_dict,
                                                date,
                                                max_symbol,
                                                trading_time_interval_dict,
                                                is_new,
                                                exchange,
                                                user,
                                                password,
                                                volume_multiple))
        process.start()
        process_list.append(process)
    
    for process in process_list:
        process.join()
    return shared_raw_data_dict

def process_feature(df):
    lag1_col_list = []
    for i in range(1, 6):
        lag1_col_list.append('BidPrice%d' % i)
        lag1_col_list.append('AskPrice%d' % i)
        lag1_col_list.append('BidVolume%d' % i)
        lag1_col_list.append('AskVolume%d' % i)
        
    diff_len1_col_list = ['Turnover', 'Volume']

    ## 1. 添加lasttick信息 (包含5档 order量价)
    for col in lag1_col_list:
        df['%s_Lag1' % col] = df[col].shift(1)

    ## 2. 添加差额信息 (包含成交量/成交金额)
    for col in diff_len1_col_list:
        df['%s_DiffLen1' % col] = df[col].diff(1)

    ## 3. 计算中间价/公允价
    for level in range(1, 6):   
        df['MidPriceLvl%d' % level] = df.apply(lambda x: cal_mid_price(x, level), axis=1)
        df['MicroPriceLvl%d' % level] = df.apply(lambda x: cal_micro_price(x, level), axis=1)

    ## 4. 计算均价
    df['VWAP'] = df.apply(lambda x: cal_vwap(x), axis=1)

    ## 5. 计算tick内 买入/卖出 成交单数
    df['MarketBuyVolume'] = df.apply(lambda x: cal_market_buy_volume(x), axis=1)
    df['MarketSellVolume'] = df.apply(lambda x: cal_market_sell_volume(x), axis=1)
    
    ## 6. 计算总买入/卖出 成交单数
    for accum_col in ['MarketBuyVolume', 'MarketSellVolume']:
        append_accum_feature(df, accum_col)
        
    ## 7. 计算未来N-tick 价格变动
    for length in [1, 20, 40, 60]:
        df['ReturnLen%d' % length] = df['MidPriceLvl1'].shift(-length) - df['MidPriceLvl1']
    return df

def append_feature(shared_feature_data_dict, shared_data_dict, key):
    daily_dict = dict()
    for session in SESSION_NAME_LIST:
        session_df = shared_data_dict[key][session]
        if len(session_df) > 0:
            session_df = process_feature(session_df)
        daily_dict[session] = session_df
    shared_feature_data_dict[key] = daily_dict

def append_feature_t(shared_raw_data_dict):
    shared_feature_data_dict=dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_list = list()
        for key in shared_raw_data_dict.keys():
            print(key)
            append_feature(shared_feature_data_dict, shared_raw_data_dict, key)            
    return shared_feature_data_dict




start_time = datetime.datetime.today() - datetime.timedelta(80)
end_time = datetime.datetime.today() - datetime.timedelta(10)
shared_raw_data_dict=get_ts_info(start_time,end_time,"rb")



shared_feature_data_dict=append_feature_t(shared_raw_data_dict)


with open("./explore/rb_raw_feature.pkl","wb") as f:
    pickle.dump(shared_feature_data_dict,f)

with open("./explore/rb_sql_feature.pkl","wb") as f:
    pickle.dump(shared_raw_data_dict,f)

