import argparse
import sys
import pandas as pd
from datetime import datetime, timedelta
import multiprocessing
import warnings
sys.path.append("/home/lishiyu/Project/tick_strategy")
from data_utils import *
from feature import *

## USAGE
## python3 get_feature_data.py -d1 20221201 -d2 20230318 -n rb

trading_time_interval_dict = {
    'night': (3600 * -3, 3600 * -1),
    'day1': (3600 * 9, 3600 * 10 + 60 * 15),
    'day2': (3600 * 10 + 60 * 30, 3600 * 11 + 60 * 30),
    'day3': (3600 * 13 + 60 * 30, 3600 * 15)
}
#tick_size_in_yuan = 1
volume_multiple = 10
exchange = 'SHFE'    
user = ''
password = ''
is_new=False

def get_ts_info(start_time,end_time,symbol_class):
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time).strftime('%Y-%m-%d')
    manager = multiprocessing.Manager()    
    shared_raw_data_dict = manager.dict()
    process_list = list()
    i = 0
    while True:
        date = (start_time + timedelta(i)).strftime('%Y-%m-%d')
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
        df['%s_Lag2' % col] = df[col].shift(2)
        df['%s_Lag3' % col] = df[col].shift(3)
        df['%s_Lag4' % col] = df[col].shift(4)
        df['%s_Lag5' % col] = df[col].shift(5)
        
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

def get_feature_data(d1, d2, n):
    import re
    file_dir = "/home/lishiyu/Project/data_process/cache/"
    files = os.listdir(file_dir)
    symbol_dict = dict()
    symbol_dict1 = dict()
    res = dict()
    for i in files:
        _tmp = re.search("\d+\.pkl", i)
        _tmp1 = re.search(n + "\d+", i)
        if _tmp is not None and _tmp1 is not None:
            symbol_dict[i[:_tmp.start()]] = i
            symbol_dict1[i[:_tmp.start()]] = _tmp1.group()
            
    for i in pd.date_range(d1, d2):
        s1 = i.strftime("%Y-%m-%d")
        s2 = s1 + "&" + n
        if s2 in symbol_dict:
            res[(s1, symbol_dict1[s2])] = pd.read_pickle(file_dir + symbol_dict[s2])
    return res


if __name__ == "__main__":
    default_end_time = (datetime.now() - timedelta(3)).strftime("%Y%m%d")
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', "--name", type=str, default="rb")
    parser.add_argument('-d1', "--start_time", type=str, default='20221201')
    parser.add_argument('-d2', '--end_time', type=str, default=default_end_time)
    args = parser.parse_args()
    start_time = pd.to_datetime(args.start_time)
    end_time = pd.to_datetime(args.end_time)
    print(start_time)
    print(end_time)    

    shared_raw_data_dict=get_ts_info(start_time,end_time,"rb")
    shared_feature_data_dict=append_feature_t(shared_raw_data_dict)
    
    for i, j in shared_feature_data_dict.items():
        filename = "&". join(i) + ".pkl"
        with open(f"/home/lishiyu/Project/data_process/cache/{filename}","wb") as f:
            pickle.dump(j,f)
        print(filename)

