from data_process.local_sql import table_ch, has_table, has_date, read_sql, client, typeinfo
from data_process.kline import table_pipeline, kline_period
import pandas as pd
import numpy as np
import math
from common.append_df import cc1, cc2
import talib
from talib.abstract import *
from data_process.talib_info import func_info, single_func, multi_func
from bin_tools import bins

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

    def get_input_columns(self):
        sql = f"""select name,type from system.columns
        where table='{self.input_table}' and database='{self.db}'"""
        self.input_col_type = read_sql(sql)

class table_kline_x(table_pipeline_pd):
    def __init__(self, close_idx, surfix, **params):
        super().__init__(**params)
        self.close_idx = close_idx
        self.surfix = surfix        
        
    def get_join_data(self):
        return read_sql(self._join_sql())
        
    def _join_sql(self):
        self.get_input_columns()
        idy = "b." + self.input_col_type["name"]. cc("^OM{0,1}R[MT]|^M{0,1}R[MT]|WAP")
        return f"""
        select a.*,{','.join(idy.tolist())}
        from {self.name} as a
        inner join {self.input_name} as b
        on a.date=b.date
        and a.time=b.time
        order by a.date,a.time
        """

class table_talib_period_whole(table_kline_x):
    def __init__(self, period, **params):
        super().__init__(**params)
        self.period = period
    
    def _get_sql(self):
        return f"""
        select
        date,
        time,
        Symbol,
        high_{self.period} as high,
        low_{self.period} as low,
        {self.close_idx} as close,
        vol_{self.period} as volume
        from {self.input_name}
        order by date,time
        """

    def transform_data(self, df):
        l = list()
        for _, _df in df.groupby("Symbol"):
            for i in range(self.period):
                __df = _df.iloc[i::self.period,:]
                __df["open"] = __df["close"]. shift(1).ffill()
                __df1 = append_techIdxBasic(__df)
                __df1.columns = self.surfix + "_" + __df1.columns
                __df = pd.concat([__df, __df1], axis=1)
                l.append(__df)
        return pd.concat(l)

class table_talib_normal(table_kline_x):
    def _get_sql(self):
        return f"""
        select
        date,
        time,
        high,
        low,
        {self.close_idx} as close,
        vol as volume
        from {self.input_name}
        order by date,time
        """
    
    def transform_data(self, df):
        df["open"] = df["close"]. shift(1).ffill()
        df1 = append_techIdxBasic(df)
        df1.columns = self.surfix + "_" + df1.columns
        df = pd.concat([df, df1], axis=1)
        return df

if __name__ == "__test__":
    TXM = table_talib_period_whole(period=600,
                                   close_idx="MID",
                                   surfix="TXM", 
                                   code=code, input_table=f"kline_whole",output_table=f"TXM_{i}_whole")

    TXM1 = table_talib_normal(
        close_idx = "MID",
        surfix="TXM", 
        code=code,
        input_table="kline_300",
        output_table="TXM_300")

    df1 = TXM1.get_data()
    df = TXM.get_data()

    gg = df1.merge(df, on=["date", "time"], how="left")
    gg[["TXM_BBANDS_middleband_y", "TXM_BBANDS_middleband_x", "date", "time"]]. sort_values(["date", "time"])


    TXM.get_join_data()    

