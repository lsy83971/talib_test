import requests
import pickle
import os
from collections import defaultdict
import pandas as pd
import numpy as np

FLOAT_TIME_COL = 'ExchTimeOffsetUs'

SIX_OCLOCK_IN_SECONDS = 6 * 3600
NIGHT_MARKET_START_TIME = -3600 * 3
DAY_START_TIME1 = 9 * 3600
DAY_START_TIME2 = 10 * 3600 + 30 * 60
DAY_START_TIME3 = 13 * 3600 + 30 * 60
DAY_BREAK_TIME1 = 10 * 3600 + 23 * 60
DAY_BREAK_TIME2 = 12 * 3600

SESSION_NAME_LIST = ['night', 'day1', 'day2', 'day3']


def clickhouse_query(query):
    return requests.post(
        'http://{}:8123?'.format('clickhouse.db.prod.highfortfunds.com'),
        data=query,
        auth=('readonly', ''),
        stream=True)


def clickhouse_query_new(query, user, password):
    return requests.post(
        'http://{}:8123?'.format('ch.db.prod.highfortfunds.com'),
        data=query,
        auth=(user, password),
        stream=True)


def get_daily_data(date, symbol):
    query = """
    select
    TradingDay
    ,Symbol
    ,NumericExchTime as ExchTimeOffsetUs
    ,Volume
    ,Turnover
    ,OpenInterest
    ,BidPrice1
    ,BidVolume1
    ,BidOrdCnt1 as BidCount1
    ,BidPrice2
    ,BidVolume2
    ,BidOrdCnt2 as BidCount2
    ,BidPrice3
    ,BidVolume3
    ,BidOrdCnt3 as BidCount3
    ,BidPrice4
    ,BidVolume4
    ,BidOrdCnt4 as BidCount4
    ,BidPrice5
    ,BidVolume5
    ,BidOrdCnt5 as BidCount5
    ,AskPrice1
    ,AskVolume1
    ,AskOrdCnt1 as AskCount1
    ,AskPrice2
    ,AskVolume2
    ,AskOrdCnt2 as AskCount2
    ,AskPrice3
    ,AskVolume3
    ,AskOrdCnt3 as AskCount3
    ,AskPrice4
    ,AskVolume4
    ,AskOrdCnt4 as AskCount4
    ,AskPrice5
    ,AskVolume5
    ,AskOrdCnt5 as AskCount5
    ,'' as LocalTimeStamp
    from futures.tick
    where TradingDay = '%s' and Symbol = '%s'
    order by NumericExchTime, MdID
    format CSVWithNames
    """ % (date, symbol)
    response = clickhouse_query(query)
    daily_data = pd.read_csv(response.raw)
    daily_data['LocalTime'] = -1
    return daily_data


def is_daily_data_empty(date, symbol):
    query = """
    select
    TradingDay
    ,Symbol
    from futures.tick
    where TradingDay = '%s' and Symbol = '%s'
    limit 1
    format CSVWithNames
    """ % (date, symbol)
    response = clickhouse_query(query)
    daily_data = pd.read_csv(response.raw)
    return len(daily_data) == 0


def get_daily_data_new(date, symbol, exchange, user, password):
    def count_server(x, server_counter):
        for server in eval(x).keys():
            server_counter[server] += 1

    query = """
    select
    TradingDay
    ,Symbol
    ,ExchTimeOffsetUs / 1000000.0 as ExchTimeOffsetUs
    ,Volume
    ,Turnover
    ,OpenInterest
    ,BidPrice1
    ,BidVolume1
    ,BidCount1
    ,BidPrice2
    ,BidVolume2
    ,BidCount2
    ,BidPrice3
    ,BidVolume3
    ,BidCount3
    ,BidPrice4
    ,BidVolume4
    ,BidCount4
    ,BidPrice5
    ,BidVolume5
    ,BidCount5
    ,AskPrice1
    ,AskVolume1
    ,AskCount1
    ,AskPrice2
    ,AskVolume2
    ,AskCount2
    ,AskPrice3
    ,AskVolume3
    ,AskCount3
    ,AskPrice4
    ,AskVolume4
    ,AskCount4
    ,AskPrice5
    ,AskVolume5
    ,AskCount5
    ,LocalTimeStamp
    from futures.tick
    where TradingDay = '%s' and Symbol = '%s.%s'
    order by ExchTimeOffsetUs, Volume
    format CSVWithNames
    """ % (date, symbol, exchange)
    response = clickhouse_query_new(query, user, password)
    daily_data = pd.read_csv(response.raw)
    daily_data['Symbol'] = symbol

    server_counter = defaultdict(int)
    _ = daily_data['LocalTimeStamp'].apply(lambda x: count_server(x, server_counter))
    max_server = ''
    max_count = -1
    for server in server_counter:
        if server_counter[server] > max_count:
            max_server = server
            max_count = server_counter[server]

    daily_data = daily_data[daily_data['LocalTimeStamp'].str.contains(max_server)]
    daily_data['LocalTime'] = daily_data['LocalTimeStamp'].apply(lambda x: eval(x)[max_server]) / 1000000.0

    return daily_data.sort_values(by=[FLOAT_TIME_COL, 'Volume', 'LocalTime'])


def is_daily_data_empty_new(date, symbol, exchange, user, password):
    query = """
    select
    TradingDay
    ,Symbol
    from futures.tick
    where TradingDay = '%s' and Symbol = '%s.%s'
    limit 1
    format CSVWithNames
    """ % (date, symbol, exchange)
    response = clickhouse_query_new(query, user, password)
    daily_data = pd.read_csv(response.raw)
    return len(daily_data) == 0


def get_symbol2volume_multiple_dict(date):
    """
    获取商品交易乘数
    """
    try:
        instrument_info_dict = requests.get('http://ceph-s3.prod.highfortfunds.com/futures/instruments/%s.json' % date).json()
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
        instrument_info_dict = requests.get('http://ceph-s3.prod.highfortfunds.com/futures/instruments/%s.json' % date).json()
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
    response = clickhouse_query(query)
    data = pd.read_csv(response.raw)
    if len(data) == 0:
        return ''
    else:
        return data['Symbol'][0]


def get_max_symbol_new(date, symbol_class, exchange, user, password):
    query = """
    select Symbol, max(Volume)
    from futures.tick
    where TradingDay = '%s' and match(Symbol, '^%s[0-9]+\.%s$')
    group by Symbol
    order by max(Volume) desc
    format CSVWithNames
    """ % (date, symbol_class, exchange)
    response = clickhouse_query_new(query, user, password)
    data = pd.read_csv(response.raw)
    if len(data) == 0:
        return ''
    else:
        # Symbol的格式为"合约.交易所"
        return data['Symbol'][0].split('.')[0]


def append_data(data_dict, date, symbol, trading_time_interval_dict, is_new=False, exchange=None,
                user=None, password=None, volume_multiple=None):
    if not is_new:
        daily_df = get_daily_data(date, symbol)
        if len(daily_df) == 0:
            return
    else:
        daily_df = get_daily_data_new(date, symbol, exchange, user, password)
        if len(daily_df) == 0:
            return

    symbol2volume_multiple_dict = get_symbol2volume_multiple_dict(date)
    if symbol in symbol2volume_multiple_dict:
        volume_multiple = symbol2volume_multiple_dict[symbol]
    elif volume_multiple is None:
        raise Exception('Not VolumeMultiple')
    daily_df['VolumeMultiple'] = volume_multiple

    # Processing for sessions
    night_df = daily_df.loc[(daily_df[FLOAT_TIME_COL] >= trading_time_interval_dict['night'][0]) &
                            (daily_df[FLOAT_TIME_COL] <= trading_time_interval_dict['night'][1])].reset_index(drop=True)
    day1_df = daily_df.loc[(daily_df[FLOAT_TIME_COL] >= trading_time_interval_dict['day1'][0]) &
                           (daily_df[FLOAT_TIME_COL] <= trading_time_interval_dict['day1'][1])].reset_index(drop=True)
    day2_df = daily_df.loc[(daily_df[FLOAT_TIME_COL] >= trading_time_interval_dict['day2'][0]) &
                           (daily_df[FLOAT_TIME_COL] <= trading_time_interval_dict['day2'][1])].reset_index(drop=True)
    day3_df = daily_df.loc[(daily_df[FLOAT_TIME_COL] >= trading_time_interval_dict['day3'][0]) &
                           (daily_df[FLOAT_TIME_COL] <= trading_time_interval_dict['day3'][1])].reset_index(drop=True)

    session_df_list = [night_df, day1_df, day2_df, day3_df]

    daily_dict = dict()
    for session_name, session_df in zip(SESSION_NAME_LIST, session_df_list):
        session_df['Session'] = session_name
        daily_dict[session_name] = session_df

    # multiprocessing.Manager.dict()实现时已经带锁了，可以保证进程安全
    data_dict[(date, symbol)] = daily_dict