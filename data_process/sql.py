import pandas as pd
import numpy as np
import math
import sys
from clickhouse_driver import Client
from pandas.api.types import infer_dtype
import re

client = Client(host='localhost', database='', user='default', password='irelia17')

def read_sql(sql):
    data, columns = client.execute(sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    df.colinfo = pd.DataFrame(columns, columns=["name", "typesql"])
    return df

def col_type(table_name):
    i1, i2 = table_name.split(".")    
    sql = f"""select name as surname, type as typesql from system.columns
    where table='{i2}' and database='{i1}';"""
    typeinfo = read_sql(sql)
    typeinfo["type"] = typeinfo["typesql"]. apply(
        lambda x:
        "integer" if "Int" in x else
        "floating" if "Float" in x else
        "boolean" if "Bool" in x else               
        "string"
    )
    assert(typeinfo.loc[typeinfo["surname"]. str.startswith("MX_"), "type"] == "string").all()
    typeinfo.loc[typeinfo["surname"]. str.startswith("MX_"), "type"] = "mixed"
    typeinfo["name"] = typeinfo["surname"]
    typeinfo.loc[typeinfo["type"] == "mixed", "name"] = \
        typeinfo.loc[typeinfo["type"] == "mixed", "surname"]. str[3:]
    typeinfo.index = typeinfo["name"]    
    return typeinfo

class table_creator:
    type_map = {"floating": "Float64",
                "integer": 'Int64',
                "boolean": "Int8",
                "string": "String",
                "mixed": "String"
                }
    
    def __init__(self, table_name):
        self.table_name = table_name
        self.check_exists()

    def check_exists(self):
        self.db, self.tb = self.table_name.split(".")    
        df = read_sql(f"""select * from system.tables
        where database='{self.db}' and name='{self.tb}'""")
        if df.shape[0] > 0:
            self.has = True
        else:
            self.has = False

    def check_type(self):
        self.typeinfo = col_type(self.table_name)

    def check_df(self, df):
        self.df_typeinfo = pd.DataFrame(
            [(i, infer_dtype(df[i])) for i in df.columns], columns=["name", "type"])
        self.df_typeinfo.index = self.df_typeinfo["name"]
        if not "id" in df.columns:
            self.df_need_id = True
            self.df_typeinfo.loc["id"] = ["id", "integer"]
        else:
            self.df_need_id = False

        self.df_typeinfo["surname"] = self.df_typeinfo.apply(
            lambda x:"MX_" + x["name"] if (not x["name"]. startswith("MX_")) \
            and (x["type"] == "mixed") else x["name"], 
            axis=1)

        self.df_typeinfo["typesql"] = self.df_typeinfo["type"].apply(
            lambda x:self.type_map.get(x, x))
        self.df_columns_info = ",". join(
            [f"`{i}` {j}" for _, (i, j) in self.df_typeinfo[["surname", "typesql"]]. iterrows()])
        
        
    def create(self, df, partition=None, orderby=None):
        if partition is None:
            partition_word = ""
        elif isinstance(partition, list):
            partition_word = f"partition by ({','.join(partition)})"
        elif isinstance(partition, str):
            partition_word = f"partition by {partition}"
        else:
            raise

        if orderby is None:
            orderby_word = "order by id"
        elif isinstance(orderby, list):
            orderby_word = f"order by ({','.join(orderby)})"
        elif isinstance(orderby, str):
            orderby_word = f"order by {orderby}"
        else:
            raise
            
            
        self.check_df(df)
        self.create_sql = f"""
        create TABLE
        {self.table_name}
        ({self.df_columns_info})
        engine=MergeTree
        {orderby_word}
        {partition_word}
        """
        client.execute(self.create_sql)

    def drop(self):
        client.execute(f"drop table {table_name}")


    def reform(self, df):
        self.check_df(df)
        df = df.copy(). reset_index(drop=True)
        if self.df_need_id:
            df["id"] = df.index
            
        mxinfo = self.df_typeinfo[self.df_typeinfo["type"] == "mixed"]
        df[mxinfo["name"]] = df[mxinfo["name"]]. fillna("").astype(str)
        df.r1(mxinfo.set_index("name")["surname"])
        return df

    def de_reform(self, df):
        self.check_type()
        d = self.typeinfo.set_index("surname")["name"]
        df.r1(d)
        df.mxcols = df.columns[df.columns. isin(d.values)]. tolist()
        return df

    def _raw_insert(self, df, table_name):
        data = df.to_dict('records')
        cols = ",". join(df.columns)
        client.execute(f"INSERT INTO {table_name} ({cols}) VALUES", data, types_check=True)

    def insert(self, df, table_name):
        df = self.reform(df)
        #self.insert_data = df
        self._raw_insert(df, table_name)

    def insert_or_create(self, df, table_name, partition=None, orderby=None):
        if self.has:
            self.insert(df, table_name)
            return
        self.create(df, partition=partition, orderby=orderby)
        self.insert(df, table_name)

    def read_sql(self, sql):
        df = read_sql(sql)
        return self.de_reform(df)


def tsq(self, table_name, partition=None, orderby=None):
    tc = table_creator(table_name)
    tc.insert_or_create(self, table_name, partition=partition, orderby=orderby)

def tsq1(self, table_name, partition=None, orderby=None):
    tc = table_creator(table_name)
    for j, i in enumerate(np.split(self, range(0, self.shape[0], 10000))):
        print(j)
        tc.insert_or_create(i, table_name, partition=partition, orderby=orderby)
    
pd.DataFrame.tsq = tsq
pd.DataFrame.tsq1 = tsq1

def rsq(sql):
    df = read_sql(sql)
    mxcols = pd.Series(df.columns[df.columns.str.startswith("MX_")]).str[3:].tolist()
    df.mxcols = mxcols
    df.columns = pd.Series(df.columns).apply(lambda x:x if not x.startswith("MX_") else x[3:])
    return df

def get_kline_tick(table_name):
    # "exch_detail",
    # "ask_exch_old",
    # "ask_exch_new",
    # "bid_exch_old",
    # "bid_exch_new",
    # "moment_exch",
    # "ask_add",
    # "ask_cancel",
    # "bid_add",
    # "bid_cancel",

    sql = f"""select
    TradingDay,
    ExchTimeOffsetUs as time,
    Session,
    tick_open as open,
    WAP as close,
    tick_high as high,
    tick_low as low,
    Volume_DiffLen1 as volume,
    Turnover_DiffLen1S as amt,
    MX_exch_detail,
    RT20,
    RT40,
    RT60,
    RM1,
    RM3,
    RM5,
    RM10,
    RM15,
    RM20,
    RM30
    from {table_name}
    order by TradingDay, ExchTimeOffsetUs
    """
    return rsq(sql)

def get_kline_period(table_name, period):
    sql = f"""
    select
    TradingDay,Session,toInt32(floor(ExchTimeOffsetUs/{period})) as period_order,
    max(tick_high) as high,
    min(tick_low) as low,
    any(tick_open) as open,
    anyLast(WAP) as close,
    any(ExchTimeOffsetUs) as time,
    sum(Volume_DiffLen1) as volume,
    sum(Turnover_DiffLen1S) as amt
    from {table_name}
    group by TradingDay,Session,toInt32(floor(ExchTimeOffsetUs/{period}))
    order by TradingDay,toInt32(floor(ExchTimeOffsetUs/{period}))
    """
    df = rsq(sql)
    return df

def get_kline(table_name="rb.detail", period=None):
    if period is None:
        return get_kline_tick(table_name)
    else:
        return get_kline_period(table_name, period)

if __name__ == "__main__":
    table_name = "tickdata.test2"
    tc = table_creator(table_name)
    tc.drop()


