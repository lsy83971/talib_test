from local_sql import table_ch, has_table, has_date, read_sql, client, typeinfo
from kline import table_pipeline
import pandas as pd
import numpy as np
import math
from common.append_df import cc1, cc2
import talib
from talib.abstract import *
from talib_info import func_info, single_func, multi_func

def fna(self):
    return self.fillna(method="ffill").fillna(method="bfill")
pd.Series.fna = fna

def append_techIdxBasic(data):
    dfs = list()
    res = dict()
    for i in single_func.index:
        if i == "MAVP":
            continue
        res[i] = eval(i)(data)
    dfs.append(pd.DataFrame(res))    

    for i in multi_func.index:
        tmp = eval(i)(data)
        tmp.columns = i + "_" + tmp.columns
        dfs.append(tmp)
        
    res_df = pd.concat(dfs, axis=1)
    for idx in [j for i in func_info[func_info["group"] == "Overlap Studies"]["output_names"] for j in i]:
        if idx not in res_df.columns:
            continue
        res_df[idx] = res_df[idx] - data["close"]

    res_df.columns = "TXB_" + res_df.columns
    return res_df

def append_MAIdx(data):
    res = dict()
    for i in [10, 30, 60, 120, 180, 240, 360, 480, 600,
              720, 960, 1200, 1800, 2400, 3600
              ]:
        res["sma_" + str(i)] = SMA(data, timeperiod=i) - data["close"]
        res["ema_" + str(i)] = EMA(data, timeperiod=i) - data["close"]
        
    res_df = pd.DataFrame(res)
    res_df.columns = "TXMA_" + res_df.columns
    return res_df

class table_pipeline_pd(table_pipeline):
    def get_data(self):
        return self.transform_data(read_sql(self._get_sql()))

    def createwith(self, df):
        self.create(typeinfo(df), orderby=["date", "time"], partitionby=None)
    
    def insert_data(self):
        if not self.input_has:
            return 0
        self.drop()
        df = self.get_data()
        self.createwith(df)
        self._raw_insert(df)
        return 1

    @staticmethod
    def transform_data(df):
        raise

    def _get_sql(self):
        raise

class table_kline_x(table_pipeline_pd):
    def _get_sql(self):
        return f"""
        select
        date,
        time,
        high,
        low,
        WAP as close,
        vol as volume
        from {self.input_name}
        order by date,time
        """
    
class table_talib_normal(table_kline_x):
    @staticmethod
    def transform_data(df):
        df["open"] = df["close"]. shift(1).ffill()
        df = cc2(df, append_techIdxBasic)
        return df

if __name__ == "__main__":
    for code in ["rb", "ru", "cu"]    :
        for i in [5, 10, 15, 20, 30, 60, 90, 150, 300]:
            TXB = table_talib_normal(code, f"kline_{i}", f"TXB_{i}")
            TXB.insert_data()
    

