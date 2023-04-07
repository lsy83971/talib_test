import requests
import pickle
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import math
from common.append_df import cc2
from bin_tools import bins

FLOAT_TIME_COL = 'ExchTimeOffsetUs'

SIX_OCLOCK_IN_SECONDS = 6 * 3600
NIGHT_MARKET_START_TIME = -3600 * 3
DAY_START_TIME1 = 9 * 3600
DAY_START_TIME2 = 10 * 3600 + 30 * 60
DAY_START_TIME3 = 13 * 3600 + 30 * 60
DAY_BREAK_TIME1 = 10 * 3600 + 23 * 60
DAY_BREAK_TIME2 = 12 * 3600

SESSION_NAME_LIST = ['day0', 'day1', 'day2', 'day3']

def read_sql(sql):
    resp = requests.post(
        'http://{}:8123?'.format('clickhouse.db.prod.highfortfunds.com'),
        data=sql,
        auth=('readonly', ''),
        stream=True)
    data = pd.read_csv(resp.raw)
    return data

# def clickhouse_query_new(query, user, password):
#     return requests.post(
#         'http://{}:8123?'.format('ch.db.prod.highfortfunds.com'),
#         data=query,
#         auth=(user, password),
#         stream=True)


def get_daily_raw_data(date, symbol):
    query = """
    select
    TradingDay as date
    ,Symbol
    ,NumericExchTime as time
    ,Volume as vol_sum
    ,Turnover as amt_sum
    ,OpenInterest
    ,BidPrice1 as BP1
    ,BidVolume1 as BV1
    ,BidPrice2 as BP2
    ,BidVolume2 as BV2
    ,BidPrice3 as BP3
    ,BidVolume3 as BV3
    ,BidPrice4 as BP4
    ,BidVolume4 as BV4
    ,BidPrice5 as BP5
    ,BidVolume5 as BV5
    ,AskPrice1 as AP1
    ,AskVolume1 as AV1
    ,AskPrice2 as AP2
    ,AskVolume2 as AV2
    ,AskPrice3 as AP3
    ,AskVolume3 as AV3
    ,AskPrice4 as AP4
    ,AskVolume4 as AV4
    ,AskPrice5 as AP5
    ,AskVolume5 as AV5
    from futures.tick
    where TradingDay = '%s' and Symbol = '%s'
    order by NumericExchTime, MdID
    format CSVWithNames
    """ % (date, symbol)
    df = read_sql(query)
    
    if "time" in df.columns:
        time = df["time"]
        session = pd.Series("", index=df.index)
        for i, (j1, j2) in trading_time_interval_dict.items():
            cond = (time >= j1) & (time <= j2)
            session.loc[cond] = i
        df["Session"] = session
    return df

# df = populate_tick(df)
# df = get_daily_raw_data("2022-12-19", "rb2305")

def populate_tick(df):
    if "time" in df.columns:
        df1 = df[["Session", "time"]]
        first = df1.drop_duplicates("Session").set_index("Session")["time"]
        last = df1.drop_duplicates("Session", keep="last").set_index("Session")["time"]

        range_list = list()
        for i in ["day0", "day1", "day2", "day3"]:
            if i not in first:
                continue
            range_list.append(np.arange(first[i], last[i] + 0.5, 0.5))
        range_idx = np.concatenate(range_list)

        df.set_index("time", inplace=True)
        df = df.reindex(range_idx)
        df.reset_index(inplace=True)
        df["FLAG_ffill_tick"] = df["Session"]. isnull().astype(int)
        df.ffill(inplace=True)
    return df
    

def get_symbol2volume_multiple_dict(date):
    """
    获取商品交易乘数
    """
    try:
        instrument_info_dict = requests.get(
            'http://ceph-s3.prod.highfortfunds.com/futures/instruments/%s.json' % date).json()
        symbol2volume_multiple_dict = dict()
        for symbol in instrument_info_dict:
            symbol2volume_multiple_dict[symbol] = instrument_info_dict[symbol]['info']['vol_mul']
        return symbol2volume_multiple_dict
    except:
        return dict()

def get_symbol2exchange_dict(date):
    """
    获取商品所属交易所
    """
    try:
        instrument_info_dict = requests.get(
            'http://ceph-s3.prod.highfortfunds.com/futures/instruments/%s.json' % date).json()
        symbol2exchange_dict = dict()
        for symbol in instrument_info_dict:
            symbol2exchange_dict[symbol] = instrument_info_dict[symbol]['info']['exch_id']
        return symbol2exchange_dict
    except:
        return dict()

def get_max_symbol(date, symbol_class):
    """
    获取最大成交量品种
    """
    query = """
    select Symbol, max(Volume)
    from futures.tick
    where TradingDay = '%s' and match(Symbol, '^%s[0-9]+$')
    group by Symbol
    order by max(Volume) desc
    format CSVWithNames
    """ % (date, symbol_class)
    data = read_sql(query)
    if len(data) == 0:
        return ''
    else:
        return data['Symbol'][0]

trading_time_interval_dict = {
    'day0': (3600 * -3, 3600 * -1),
    'day1': (3600 * 9, 3600 * 10 + 60 * 15),
    'day2': (3600 * 10 + 60 * 30, 3600 * 11 + 60 * 30),
    'day3': (3600 * 13 + 60 * 30, 3600 * 15)
}

def append_basic_feature(df):
    res = dict()
    # 1, populate 
    # 2. mult
    date = df["date"]. iloc[0]
    symbol = df["Symbol"]. iloc[0]    
    v_mult = get_symbol2volume_multiple_dict(date)[symbol]
    res["vMult"] = pd.Series(v_mult, index=df.index)

    # 3. vol amt
    res["vol"] = df["vol_sum"]. diff()
    res["vol"]. iloc[0] = df["vol_sum"]. iloc[0]
    res["amt"] = df["amt_sum"]. diff() / v_mult
    res["amt"]. iloc[0] = df["amt_sum"]. iloc[0] / v_mult
    res["VWAP"] = (res["amt"] / res["vol"]).ffill()
    res["WAP"] = (((df["AP1"] * df["BV1"]) + (df["BP1"] * df["AV1"])) / (df["AV1"] + df["BV1"])).ffill()

    # 4. lag1
    # TODO may be delete day2
    init_idx = df["Session"]. drop_duplicates().index
    for i in df.columns[df.columns.str.contains("AP|BP")]:
        res[i + "_last"] = df[i]. shift(1)
        res[i + "_last"]. loc[init_idx] = df[i]. loc[init_idx]
        
    for i in df.columns[df.columns.str.contains("AV|BV")]:
        res[i + "_last"] = df[i]. shift(1)
        res[i + "_last"]. loc[init_idx] = 0
    return res

def get_daily_data(date, symbol):
    df = populate_tick(get_daily_raw_data(date, symbol))
    df = cc2(df, append_basic_feature)
    for i in df.cc("^AP|^BP|^AV|^BV"):
        df[i] = df[i]. astype(int)
    return df

def get_daily_max(date, code):
    symbol = get_max_symbol(date, code)
    if symbol == "":
        return None, ""
    df = get_daily_data(date, symbol)
    return df, symbol

if __name__ == "__main__":
    date = "2022-12-19"
    symbol = "rb2305"
    df0 = get_daily_raw_data("2022-12-19", "rb2305")
    df1 = get_daily_data("2022-12-19", "rb2305")
    df2 = get_daily_max("2022-12-19", "rb")[0]


