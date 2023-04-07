from local_sql import table_ch, has_table, has_date, read_sql, client
import pandas as pd
import numpy as np

class table_pipeline(table_ch):
    orderby = "date,time"
    def __init__(self, code, input_table, output_table):
        self.code = code
        self.db = code
        self.table = output_table
        self.input_table = input_table
        self.name = self.db + "." + self.table
        self.input_name = self.db + "." + self.input_table

    @property
    def input_has(self):
        return has_table(self.input_name)
    
    def input_has_date(self, date):
        return has_date(self.input_name, date)

    def insert_date(self, date):
        if not self.input_has_date(date):
            return 0
        if self.has_date(date):
            return 2

        if not self.has:
            need_create = True
        else:
            need_create = False
        print(date, "need_create:", need_create)
        
        if need_create:
            init_str = f"""
            CREATE TABLE {self.name}
            engine=MergeTree
            order by ({self.orderby})
            partition by date AS           
            """
        else:
            init_str = f"""
            INSERT INTO {self.name} 
            """
            
        _sql = init_str + self._raw_insert_date(date)
        client.execute(_sql)
        return 1

    def _raw_insert_date(self, date):
        raise
        
    def insert_dates(self, d1, d2):
        for i in pd.date_range(d1, d2):
            date = i.strftime("%Y-%m-%d")
            self.insert_date(date)

            
class table_kline_tick(table_pipeline):
    def __init__(self, code):
        super().__init__(input_table="tickdata",
                         output_table="kline_tick",
                         code=code
                         )
        
    def _raw_insert_date(self, date):
        _sql = f"""
        SELECT date,Session,time,vMult,Symbol,
        case when vol=0 then null
        else arrayMax(mapKeys(mapFilValuePos(D_exch))) end as high,
        case when vol=0 then null
        else arrayMin(mapKeys(mapFilValuePos(D_exch))) end as low,
        VWAP,
        WAP,
        vol,
        amt,
        D_exch
        FROM (select D_exch,VWAP,WAP,vol,amt,date,time,Session,vMult,Symbol
        from {self.input_name} where date='{date}')
        """
        return _sql

    def insert_dates(self, d1, d2):
        super().insert_dates(d1, d2)
        self.mater("ORT1 Float64", """
        last_value(VWAP) over (ORDER BY date,time ASC Rows BETWEEN current row AND 1 following)
        """)

class table_kline_period(table_pipeline):
    orderby = "date,start_time"      
    def __init__(self, code, period):
        assert isinstance(period, int)
        assert period >= 1
        assert period <= 300
        assert 900 % period == 0

        self.period = period
        super().__init__(input_table="kline_tick",
                         output_table="kline_" + str(period),
                         code=code
                         )
    
    def _raw_insert_date(self, date):
        _sql = f"""
        select
        date,
        any(Session) as Session,
        any(vMult) as vMult,
        any(Symbol) as Symbol,
        max(high) as high,
        min(low) as low,
        anyLast(WAP) as end_WAP,
        anyLast(VWAP) as end_VWAP,
        anyLast(time) as end_time,
        any(time) as start_time,
        sum(vol) as vol,
        sum(amt) as amt
        from {self.input_name}
        where date='{date}'
        group by (date,toInt32(floor((time-0.5)/{self.period})))
        """
        return _sql

########################

if __name__ == "__main__":
    tk0 = table_kline_tick("rb")
    tk0.drop()
    tk0.insert_dates("2022-11-05", "2022-11-10")


gg = read_sql("""SELECT VWAP,last_value(VWAP) over
(ORDER BY date,time ASC Rows BETWEEN current row AND 5 following)
from rb.kline_tick where date='2022-11-10'
""")
gg
gg
_sql = """
SELECT
date,
VWAP,
last_value(VWAP) over
(ORDER BY date,time ASC Rows BETWEEN current row AND 3 following) AS v
from rb.tickdata
"""
gg = read_sql(_sql)

#(gg["VWAP"]. shift( -3) - gg["v"]).value_counts()
read_sql("""SELECT WAP,AP1 as v1 from rb.tickdata order by date,time""")["WAP"]. iloc[0]


    tkp = table_kline_period("rb", 5)
    tkp.drop()
    tkp.insert_dates("2022-11-09", "2022-12-31")

    tkp = table_kline_period("rb", 10)
    tkp.drop()
    tkp.insert_dates("2022-11-09", "2022-12-31")


    read_sql("select * from rb.kline_10")


