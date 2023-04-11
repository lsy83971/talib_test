import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from clickhouse_driver import Client
import re
#from importlib import reload
#import remote_sql
#reload(remote_sql)
from remote_sql import get_daily_max
from common.append_df import cc2
from bin_tools import bins
import sys
import multiprocessing
from multiprocessing import Pool

client = Client(host='localhost', database='', user='default', password='irelia17')

def read_sql(sql):
    data, columns = client.execute(sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    df.colinfo = pd.DataFrame(columns, columns=["name", "typesql"])
    return df

exists_func_names = read_sql(f"select name from system.functions")["name"]. tolist()
for i, j in [
        ("dict_neg_key", "(x) -> mapApply((k, v) -> (-k, v),x);"), 
        ("reverse_map", "(x)-> mapFromArrays(reverse(mapKeys(x)),reverse(mapValues(x)));"), 
        ("clip_map", "(x,b1,b2)-> mapFilter((k, v) -> ((k>=b1) and (k<=b2)),x);"), 
        ("clipMapLeft", "(x,b1)-> mapFilter((k, v) -> (k>=b1),x);"), 
        ("clipMapRight", "(x,b2)-> mapFilter((k, v) -> (k<=b2),x);"), 
        ("mapFilValuePos", "(x)-> mapFilter((k, v) -> (v>0),x);"), 
        ("mapFilValueNeg", "(x)-> mapFilter((k, v) -> (v<0),x);"), 
        ("mapValueNeg", "(x)-> mapApply((k, v) -> (k, -v),x);")]:
    if i not in exists_func_names:
        print(f"create function {i}")
        client.execute(f"""CREATE FUNCTION {i} as {j}""")
    
type_map = {
    "int64": "Int64",
    "int32": "Int64",
    "integer": "Int64",
    "float64": "Float64",
    "float32": "Float32",
    "floating": "Float64",
    "bool": "Bool",
    "boolean": "Bool",
    "string": "String",
}

def typeinfo(df, add=None):
    d = pd.Series({i:infer_dtype(df[i]) for i in df.columns}).map(type_map)
    d = d[d != "mixed"]. to_dict()
    if add is not None:
        d.update(add)
    try:
        assert (~df.columns.isin(d.keys())).sum() == 0
    except:
        print(df.columns[(~df.columns.isin(d.keys()))])
    return pd.Series(d)

def has_db(db):
    return (read_sql(f"select * from system.databases where name='{db}'").shape[0] > 0)

def has_table(name):
    db, table = name.split(".")
    df = read_sql(f"""select * from system.tables
    where database='{db}' and name='{table}'""")
    if df.shape[0] > 0:
        return True
    else:
        return False

def has_date(name, date):
    if not has_table(name):
        return False
    df = read_sql(f"""select * from {name}
    where date='{date}' limit 1
    """)
    if df.shape[0] > 0:
        return True
    else:
        return False
    
class table_ch:
    def __init__(self, name):
        self.name = name
        self.db, self.table = name.split(".")

    def mater(self, col, expr):
        self.get_columns()        
        colnames = self.col_type["name"]. tolist()
        if col.split(" ")[0] in colnames:
            return 0
        _sql = f"""
        alter table {self.name} add column {col}
        MATERIALIZED
        {expr}
        """
        print("MATERIALIZED table:")
        print(_sql)
        client.execute(_sql)

    @property
    def has(self):
        return has_table(self.name)

    def has_date(self, date):
        return has_date(self.name, date)
        
    @property
    def has_db(self):
        return has_db(self.db)

    def get_columns(self):
        sql = f"""select name,type from system.columns
        where table='{self.table}' and database='{self.db}'"""
        self.col_type = read_sql(sql)

    def create_db(self):
        if not self.has_db:
            client.execute(f"""create database {self.db}""")

    def create(self, type_dict, orderby, partitionby=None):
        col_type_str = ",". join([f"`{i}` {j}" for i, j in type_dict.items()])
        
        if isinstance(orderby, list):
            orderby_str = f"order by ({','.join(orderby)})"
        elif isinstance(orderby, str):
            orderby_str = f"order by {orderby}"
        else:
            raise

        if partitionby is None:
            partitionby_str = ""
        elif isinstance(partitionby, list):
            partitionby_str = f"partition by ({','.join(partitionby)})"
        elif isinstance(partitionby, str):
            partitionby_str = f"partition by {partitionby}"
        else:
            raise

        self.create_db()

        self.create_sql =f"""
        create TABLE
        {self.name}
        ({col_type_str})
        engine=MergeTree
        {orderby_str}
        {partitionby_str}
        """
        client.execute(self.create_sql)

    def drop(self):
        if self.has:
            client.execute(f"drop table {self.name}")

    def _raw_insert(self, df):
        pool = Pool(processes=12)
        cols = ",". join(df.columns)        
        for j, i in enumerate(np.split(df, range(100000, df.shape[0], 100000))):
            print(f"insert part {j}")
            pool.apply_async(_insert_df, args=(i, self.name, cols))
            #data = i.to_dict('records')
            #client.execute(f"INSERT INTO {self.name} ({cols}) VALUES", data, types_check=True)
        print("join start")
        pool.close()
        pool.join()
        print("join end")

def _insert_df(data, name, cols):
    data = data.to_dict('records')
    client = Client(host='localhost', database='', user='default', password='irelia17')
    client.execute(f"INSERT INTO {name} ({cols}) VALUES", data, types_check=True)

class exch_detail(table_ch):
    
    def __init__(self, code):
        self.table = "tickdata"
        self.db = code
        self.code = code
        self.name = code + "." + self.table
    
    def create(self, type_dict, orderby, partitionby=None):
        super().create(type_dict, orderby, partitionby)
        self.addcol_mater()

    def addcol_mater(self):
        self.addcol_pvdict()
        self.addcol_cumsum()
        self.addcol_bound()
        self.addcol_exchdetail()
        self.addcol_orderdetail()
    
    def addcol_pvdict(self):
        for i1 in ["ask", "bid"]:
            for i2 in ["", "_last"]:
                colname = "D_" + i1 + i2
                
                    
                pairs = list()
                for i3 in range(1, 6):
                    pairs.append(f"{i1[0].upper()}P{i3}{i2}")
                    pairs.append(f"{i1[0].upper()}V{i3}{i2}")
                pairs = ",". join(pairs)
                
                if i1 == "bid":
                    reverse_str1 = "reverse_map("
                    reverse_str2 = ")"
                else:
                    reverse_str1 = ""
                    reverse_str2 = ""
                
                if i2 == "":
                    self.mater(f"{colname} Map(Int64, Int64)",
                               f"{reverse_str1}mapPopulateSeries(map({pairs})){reverse_str2}"
                               )
                if i2 == "_last":
                    self.mater(f"{colname} Map(Int64, Int64)",
                               f"""
                               case when (BP1_last = -1) then map()
                               else
                               {reverse_str1}mapPopulateSeries(map({pairs})){reverse_str2} end;                               
                               """
                               )
                    
        for i1 in ["ask", "bid"]:
            colname = "D_" + i1 + "_diff"
            table = "D_" + i1
            table_last = "D_" + i1 + "_last"
            if i1 == "bid":
                reverse_str1 = "reverse_map("
                reverse_str2 = ")"
            else:
                reverse_str1 = ""
                reverse_str2 = ""

            self.mater(f"{colname} Map(Int64, Int64)",
                       f"{reverse_str1}mapSubtract({table_last},{table}){reverse_str2}"
                       )
            
    def addcol_cumsum(self):
        for col in ["D_ask", "D_ask_last", "D_ask_diff",
                    "D_bid", "D_bid_last", "D_bid_diff"
                    ]:

            if "_diff" in col:
                v_col = f"arrayMap((x)->max2(0,x),mapValues({col}))"
            else:
                v_col = f"mapValues({col})"

            self.mater(f"{col}_cumsum Map(Int64,Int64)",
                       f"mapFromArrays(mapKeys({col}),arrayCumSum({v_col}))")

        for col in ["ask", "bid"]:
            D_col = "D_" + col + "_last"
            D_a_col = "D_" + col + "_last_acumsum"
            self.mater(f"{D_a_col} Map(Int64,Int64)",
                       f"""
                       mapFromArrays(
                       mapKeys({D_col}),
                       arrayCumSum(
                       arrayMap((x,y)->(x*y),mapKeys({D_col}),mapValues({D_col}))
                       )
                       )
                       """)
                       
    def addcol_bound(self):
        ask_start = "cast(min2(max2(AP1_last, AP1),min2(AP5_last,AP5)) as Int64)"
        ask_end = f"cast(min2(ask_start+4,min2(AP5_last,AP5)+1) as Int64)"
        
        self.mater("ask_start Int64",
                   f"{ask_start}")
                            
        self.mater("ask_bound Int64",
                   f"""
                   ask_start+arrayCount(
                   (x) -> (D_ask_last_cumsum[x]<=vol*2+150),
                   range(ask_start,{ask_end}))""")


        bid_start = "cast(max2(min2(BP1_last, BP1),max2(BP5_last,BP5)) as Int64)"
        bid_end = f"cast(max2(bid_start-4,max2(BP5_last,BP5)-1) as Int64)"

        self.mater("bid_start Int64",f"{bid_start}")
        self.mater("bid_bound Int64",
                   f"""
                   bid_start-arrayCount(
                   (x) -> (D_bid_last_cumsum[x]<=vol*2+150),
                   range(bid_start,{bid_end},-1))""")

    def addcol_exchdetail(self):
        self.mater("D_exch Map(Int64,Int64)",
                   """
                   lsy_exch_detail(
                   vol,
                   amt,
                   D_ask_last,
                   D_bid_last,
                   D_ask_diff,
                   D_bid_diff,
                   D_ask_cumsum,
                   D_ask_last_cumsum,
                   D_ask_diff_cumsum,
                   D_bid_cumsum,
                   D_bid_last_cumsum,
                   D_bid_diff_cumsum,
                   D_ask_last_acumsum,
                   D_bid_last_acumsum,
                   ask_bound,
                   bid_bound)
                   """
                   )

    def addcol_orderdetail(self):
        self.mater("D_ask_change Map(Int64,Int64)",
                   "clipMapRight(mapSubtract(mapAdd(D_exch,D_ask),D_ask_last),min2(AP5,AP5_last))")

        self.mater("D_bid_change Map(Int64,Int64)", 
                   "clipMapLeft(mapSubtract(mapAdd(D_exch,D_bid),D_bid_last),max2(BP5,BP5_last))")
        
        self.mater("D_ask_add Map(Int64,Int64)", "mapFilValuePos(D_ask_change)")
        self.mater("D_ask_cancel Map(Int64,Int64)", "mapValueNeg(mapFilValueNeg(D_ask_change))")
        self.mater("D_ask_exch_new Map(Int64,Int64)", "mapFilValuePos(mapSubtract(D_exch,D_ask_last))")
        self.mater("D_ask_exch_old Map(Int64,Int64)", "mapSubtract(D_exch,D_ask_exch_new)")

        self.mater("D_bid_add Map(Int64,Int64)", "mapFilValuePos(D_bid_change)")
        self.mater("D_bid_cancel Map(Int64,Int64)", "mapValueNeg(mapFilValueNeg(D_bid_change))")
        self.mater("D_bid_exch_new Map(Int64,Int64)", "mapFilValuePos(mapSubtract(D_exch,D_bid_last))")
        self.mater("D_bid_exch_old Map(Int64,Int64)", "mapSubtract(D_exch,D_bid_exch_new)")

        self.mater("D_exch_moment Map(Int64,Int64)", "mapSubtract(D_ask_exch_new,D_bid_exch_old)")

    def insert_dates(self, d1, d2):
        if not self.has:
            need_create = True
        else:
            need_create = False
            self.addcol_mater()
            
        for i in pd.date_range(d1, d2):
            date = i.strftime("%Y-%m-%d")
            if self.has_date(date):
                continue

            df, symbol = get_daily_max(date, self.code)
            if df is None:
                continue

            if df.shape[0] < 1000:
                continue

            if need_create:
                self.create(typeinfo(df), orderby=["date", "time"], partitionby="date")
                need_create = False
            print(f"insert symbol: {symbol}, date: {date}, lines: {df.shape[0]}")
            self._raw_insert(df)

if __name__ == "__main__":
    sb = read_sql("select D_exch,amt from rb.tickdata")
    #sb["D_exch"]. apply(len).value_counts().sort_index()
    d1 = "20221105"
    d2 = "20230404"
    tick_table = exch_detail("rb")
    self = tick_table
    tick_table.insert_dates(d1, d2)

    tick_table = exch_detail("ru")
    tick_table.insert_dates(d1, d2)
    
    #d1 = "20221105"
    #d2 = "20221110"
    
    tick_table = exch_detail("rb")
    #tick_table.drop()
    #tick_table.drop()
    tick_table.insert_dates(d1, d2)
    
    #read_sql("select D_exch,VWAP,WAP from rb.tickdata limit 100")
    
