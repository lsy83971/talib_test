import os
import pandas as pd
import numpy as np
import requests

pd.set_option("display.max_rows", 500)

def clq(query):
    response = requests.post(
        'http://{}:8123?'.format('clickhouse.db.prod.highfortfunds.com'),
        data=query + " format CSVWithNames",
        auth=('readonly', ''),
        stream=True)
    df = pd.read_csv(response.raw)
    #df['LocalTime'] = -1
    return df


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














s1 = get_daily_data("2023-02-22", "IC2304")
s1




df = clickhouse_query("""select TradingDay,Symbol
    from futures.tick
    limit 100
    format CSVWithNames
""")


df = clq("""select *
    from information_schema.tables
""")

df = clq("""select *
    from information_schema.columns
""")

df1 = df[df["table_schema"] == "stock"]
df1["table_name"]. value_counts()




df_tick = clq("""select *
from stock.tick
where Symbol='000001.SZ'
limit 100000
""")


df_tick.shape



df_tick.iloc[0]


df_tick.shape
(df_tick["BidPrice1"] > 0).mean()
(df_tick["BidPrice1"] > 0).mean()




