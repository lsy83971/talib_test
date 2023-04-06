import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from clickhouse_driver import Client
import re

#from importlib import reload
#import remote_sql
#reload(remote_sql)
from remote_sql import get_daily_data, append_basic_feature, get_max_symbol
from append_df import cc2
import bins

client = Client(host='localhost', database='', user='default', password='irelia17')

# client.execute("""CREATE FUNCTION dict_neg_key as (x)
# -> mapApply((k, v) -> (-k, v),x);""")
# client.execute("""CREATE FUNCTION reverse_map as (x)
# -> mapFromArrays(reverse(mapKeys(x)),reverse(mapValues(x)));""")
# client.execute("""CREATE FUNCTION clip_map as (x,b1,b2)
# -> mapFilter((k, v) -> ((k>=b1) and (k<=b2)),x);""")
# client.execute("""CREATE FUNCTION clipMapLeft as (x,b1)
# -> mapFilter((k, v) -> (k>=b1),x);""")
# client.execute("""CREATE FUNCTION clipMapRight as (x,b2)
# -> mapFilter((k, v) -> (k<=b2),x);""")

# client.execute("""CREATE FUNCTION mapFilValuePos as (x)
# -> mapFilter((k, v) -> (v>0),x);""")
# client.execute("""CREATE FUNCTION mapFilValueNeg as (x)
# -> mapFilter((k, v) -> (v<0),x);""")
# client.execute("""CREATE FUNCTION mapValueNeg as (x)
# -> mapApply((k, v) -> (k, -v),x);""")

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

def read_sql(sql):
    data, columns = client.execute(sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    df.colinfo = pd.DataFrame(columns, columns=["name", "typesql"])
    return df

class table_ch:
    def __init__(self, name):
        self.name = name
        self.db, self.table = name.split(".")

    def mater(self, col, expr):
        _sql = f"""
        alter table {self.name} add column {col}
        MATERIALIZED
        {expr}
        """
        client.execute(_sql)

    @property
    def has(self):
        df = read_sql(f"""select * from system.tables
        where database='{self.db}' and name='{self.table}'""")
        if df.shape[0] > 0:
            return True
        else:
            return False

    def has_date(self, date):
        if not self.has:
            return False
        df = read_sql(f"""select * from {self.name}
        where date='{date}' limit 1
        """)
        if df.shape[0] > 0:
            return True
        else:
            return False
        

    def get_columns(self):
        sql = f"""select name,type from system.columns
        where table='{self.table}' and database='{self.db}'"""
        self.col_type = read_sql(sql)

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
        data = df.to_dict('records')
        cols = ",". join(df.columns)
        client.execute(f"INSERT INTO {self.name} ({cols}) VALUES", data, types_check=True)

class exch_detail(table_ch):
    def create(self, type_dict, orderby, partitionby=None):
        super().create(type_dict, orderby, partitionby)
        self.addcol_pvdict()
        self.addcol_cumsum()
        self.addcol_bound()
        self.addcol_exchdetail()
        self.addcol_orderdetail()
    
    def addcol_pvdict(self):
        self.get_columns()
        names = self.col_type["name"]
        for i1 in ["ask", "bid"]:
            for i2 in ["", "_last"]:
                colname = "D_" + i1 + i2
                
                if colname in names:
                    motion = "modify"
                else:
                    motion = "add"
                    
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
                    _sql = f"""alter table {self.name} {motion} column
                    {colname} Map(Int64, Int64) MATERIALIZED
                    {reverse_str1}mapPopulateSeries(map({pairs})){reverse_str2};
                    """
                if i2 == "_last":
                    _sql = f"""alter table {self.name} {motion} column
                    {colname} Map(Int64, Int64) MATERIALIZED
                    case when (BP1_last = -1) then map()
                    else
                    {reverse_str1}mapPopulateSeries(map({pairs})){reverse_str2} end;
                    """
                client.execute(_sql)

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
            
            _sql = f"""alter table {self.name} {motion} column
            {colname} Map(Int64, Int64) MATERIALIZED
            {reverse_str1}mapSubtract({table_last},{table}){reverse_str2}            
            """
            client.execute(_sql)

    def addcol_cumsum(self):
        for col in ["D_ask", "D_ask_last", "D_ask_diff",
                    "D_bid", "D_bid_last", "D_bid_diff"
                    ]:

            if "_diff" in col:
                v_col = f"arrayMap((x)->max2(0,x),mapValues({col}))"
            else:
                v_col = f"mapValues({col})"                
                
            _sql = f"""alter table {self.name} add column
            {col}_cumsum Map(Int64,Int64) MATERIALIZED
            mapFromArrays(mapKeys({col}),arrayCumSum({v_col}))
            """
            client.execute(_sql)


        for col in ["ask", "bid"]:
            D_col = "D_" + col + "_last"
            D_a_col = "D_" + col + "_last_acumsum"
            
            _sql = f"""alter table {self.name} add column
            {D_a_col} Map(Int64,Int64) MATERIALIZED
            mapFromArrays(
            mapKeys({D_col}),
            arrayCumSum(
             arrayMap((x,y)->(x*y),mapKeys({D_col}),mapValues({D_col}))
            )
            )
            """
            client.execute(_sql)

    def addcol_bound(self):
        ask_start = "cast(min2(max2(AP1_last, AP1),min2(AP5_last,AP5)) as Int64)"
        ask_end = f"cast(min2(ask_start+3,min2(AP5_last,AP5)+1) as Int64)"
        _sql = f"""
        alter table {self.name} add column ask_start
        Int64 MATERIALIZED {ask_start}
        """
        client.execute(_sql)
        _sql = f"""
        alter table {self.name} add column ask_bound
        Int64 MATERIALIZED
        ask_start+arrayCount(
        (x) -> ((D_ask_cumsum[x]<=vol/2+50) and (D_ask_last_cumsum[x]<=vol)),
        range(ask_start,{ask_end})
        
        )
        """
        client.execute(_sql)

        bid_start = "cast(max2(min2(BP1_last, BP1),max2(BP5_last,BP5)) as Int64)"
        bid_end = f"cast(max2(bid_start-3,max2(BP5_last,BP5)-1) as Int64)"
        _sql = f"""
        alter table {self.name} add column bid_start
        Int64 MATERIALIZED {bid_start}
        """
        client.execute(_sql)
        _sql = f"""
        alter table {self.name} add column bid_bound
        Int64 MATERIALIZED
        bid_start-arrayCount(
        (x) -> ((D_bid_cumsum[x]<=vol/2+50) and (D_bid_last_cumsum[x]<=vol)),
        range(bid_start,{bid_end},-1)
        )
        """
        client.execute(_sql)

    def addcol_exchdetail(self):
        _sql = f"""
        alter table {self.name} add column D_exch
        Map(Int64,Int64) MATERIALIZED
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
        client.execute(_sql)

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

def insert_tick_table(d1, d2, code, tb_name):
    if not tick_table.has:
        need_create = True
    else:
        need_create = False

    for i in pd.date_range(d1, d2):
        date = i.strftime("%Y-%m-%d")
        if tick_table.has_date(date):
            continue

        symbol = get_max_symbol(date, code)
        if symbol == "":
            continue

        df = get_daily_data(date, symbol)
        df = cc2(df, append_basic_feature)
        if df.shape[0] < 1000:
            continue

        if need_create:
            tick_table.create(typeinfo(df), orderby=["date", "time"], partitionby="date")
            need_create = False
        print(f"insert symbol: {symbol}, date: {date}")
        tick_table._raw_insert(df)

if __name__ == "__main__":
    d1 = "20221201"
    d2 = "20230404"
    code = "rb"
    tb_name = "rb.tickdata"
    insert_tick_table(d1, d2, code, tb_name)
