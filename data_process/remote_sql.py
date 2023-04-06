import requests
import pickle
import os
from collections import defaultdict
import pandas as pd
import numpy as np
import math

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


def get_daily_data(date, symbol):
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
    return read_sql(query)

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

    # 1. mult
    date = df["date"]. iloc[0]
    symbol = df["Symbol"]. iloc[0]    
    v_mult = get_symbol2volume_multiple_dict(date)[symbol]
    res["vMult"] = pd.Series(v_mult, index=df.index)

    # 2. session
    time = df["time"]
    session = pd.Series("", index=df.index)
    for i, (j1, j2) in trading_time_interval_dict.items():
        cond = (time >= j1) & (time <= j2)
        session.loc[cond] = i
    res["Session"] = session

    # 3. vol amt
    res["vol"] = df["vol_sum"]. diff()
    res["vol"]. iloc[0] = df["vol_sum"]. iloc[0]
    res["amt"] = df["amt_sum"]. diff() / v_mult
    res["amt"]. iloc[0] = df["amt_sum"]. iloc[0] / v_mult
    res["VWAP"] = res["amt"] / res["vol"]

    # 4. lag1
    # TODO may be delete day2
    init_idx = res["Session"]. drop_duplicates().index
    for i in df.columns[df.columns.str.contains("AP|BP")]:
        res[i + "_last"] = df[i]. shift(1)
        res[i + "_last"]. loc[init_idx] = df[i]. loc[init_idx]
        res[i + "_last"] = res[i + "_last"]. astype(int)

    for i in df.columns[df.columns.str.contains("AV|BV")]:
        res[i + "_last"] = df[i]. shift(1)
        res[i + "_last"]. loc[init_idx] = 0
        res[i + "_last"] = res[i + "_last"]. astype(int)
        
    return res


if __name__ == "__main__":
    df = get_daily_data(date, symbol)
    df = cc2(df, append_basic_feature)
    df["test"] = True
