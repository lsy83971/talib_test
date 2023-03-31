import numpy as np
import pandas as pd

EPSILON = 1e-8


def is_valid_price(price):
    return price > 0 and price < 1000 * 10000


def append_lag_feature(df, col, lag):
    df['%s_Lag%d' % (col, lag)] = df[col].shift(lag)


def append_accum_feature(df, col):
    df['Accum_%s' % col] = df[col].cumsum()


def append_diff_feature(df, col, length):
    df['%s_DiffLen%d' % (col, length)] = df[col].diff(length)


def cal_mid_price(snapshot, level=1):
    """
    ask_n 与 bid_n 均价
    """
    if not is_valid_price(snapshot['AskPrice1']):
        return snapshot['BidPrice1']
    elif not is_valid_price(snapshot['BidPrice1']):
        return snapshot['AskPrice1']
    else:
        mid_price = (snapshot['BidPrice%d' % level] +
                     snapshot['AskPrice%d' % level]) / 2.0
        return mid_price


def cal_micro_price(snapshot, level=1):
    """
    公允价格
    """
    if not is_valid_price(snapshot['AskPrice1']):
        return snapshot['BidPrice1']
    elif not is_valid_price(snapshot['BidPrice1']):
        return snapshot['AskPrice1']
    else:
        if snapshot['AskVolume%d' % level] + snapshot['BidVolume%d' % level] == 0:
            return (snapshot['AskPrice%d' % level] + snapshot['BidPrice%d' % level]) / 2.0
        micro_price = (snapshot['BidPrice%d' % level] * snapshot['AskVolume%d' % level] +
                       snapshot['AskPrice%d' % level] * snapshot['BidVolume%d' % level]) * 1.0 / (
                           snapshot['AskVolume%d' % level] + snapshot['BidVolume%d' % level])
        return micro_price


def cal_vwap(snapshot):
    """
    成交均价
    """
    if snapshot['Volume_DiffLen1'] > 0:
        return snapshot['Turnover_DiffLen1'] / snapshot['Volume_DiffLen1'] / snapshot['VolumeMultiple']
    elif snapshot['Volume_DiffLen1'] < 0:
        return -1
    else:
        return np.nan


def cal_market_buy_volume(snapshot):
    """
    向上买入成交数量
    """
    if snapshot['AskPrice1_Lag1'] == snapshot['BidPrice1_Lag1']:
        return np.nan
    market_buy_volume = (snapshot['Turnover_DiffLen1'] - snapshot['Volume_DiffLen1'] * snapshot['VolumeMultiple'] *
                         snapshot['BidPrice1_Lag1']) / (snapshot['AskPrice1_Lag1'] - snapshot['BidPrice1_Lag1']) / (
        snapshot['VolumeMultiple'])
    return min(max(market_buy_volume, 0), snapshot['Volume_DiffLen1'])


def cal_market_sell_volume(snapshot):
    """
    向下卖出成交数量
    """
    return snapshot['Volume_DiffLen1'] - snapshot['MarketBuyVolume']
