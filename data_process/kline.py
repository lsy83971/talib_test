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

    def insert_data(self):
        if not self.input_has:
            return 0
        self.drop()
        init_str = f"""
        CREATE TABLE {self.name}
        engine=MergeTree
        order by ({self.orderby})
        -- partition by date
        AS           
        """
        client.execute(init_str + self._insert_sql())
        return 1

    def _insert_sql(self):
        raise
            
class table_kline_tick(table_pipeline):
    def __init__(self, code):
        super().__init__(input_table="tickdata",
                         output_table="kline_tick",
                         code=code
                         )
        
    def _insert_sql(self):
        l_str = list()
        for i, j in [
                ("RM30", 30 * 120, ), 
                ("RM20", 20 * 120, ), 
                ("RM15", 15 * 120, ), 
                ("RM10", 10 * 120, ), 
                ("RM5", 5 * 120, ), 
                ("RM3", 3 * 120, ), 
                ("RM2", 2 * 120, ), 
                ("RM1", 1 * 120, ), 
                ("RT60", 60, ), 
                ("RT30", 30, ), 
                ("RT20", 20, ), 
                ("RT10", 10, ), 
                ("RT5", 5, ), 
        ]:
            for window, surfix in [("w1", "O"), ("w2", "")]:
                l_str.append(f"""
                (case when (nth_value(VWAP,{j+1}) over {window})=0 then null
                else nth_value(VWAP,{j+1}) over {window} END)-VWAP as {surfix}{i}
                """)
                
        
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
        D_exch,
        {",".join(l_str)}
        
        FROM {self.input_name}
        WINDOW
        w1 AS (ORDER BY date,time asc Rows BETWEEN current row AND 3600 following),
        w2 AS (PARTITION BY date ORDER BY time asc Rows BETWEEN current row AND 3600 following)
        """
        #print(_sql)
        return _sql

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
    
    def _insert_sql(self):
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
        
        anyLast(ORM30) as ORM30,
        anyLast(ORM20) as ORM20,
        anyLast(ORM15) as ORM15,
        anyLast(ORM10) as ORM10,
        anyLast(ORM5) as ORM5,
        anyLast(ORM3) as ORM3,
        anyLast(ORM2) as ORM2,
        anyLast(ORM1) as ORM1,
        anyLast(ORT60) as ORT60,
        anyLast(ORT30) as ORT30,
        anyLast(ORT20) as ORT20,
        anyLast(ORT10) as ORT10,
        anyLast(ORT5) as ORT5,

        anyLast(RM30) as RM30,
        anyLast(RM20) as RM20,
        anyLast(RM15) as RM15,
        anyLast(RM10) as RM10,
        anyLast(RM5) as RM5,
        anyLast(RM3) as RM3,
        anyLast(RM2) as RM2,
        anyLast(RM1) as RM1,
        anyLast(RT60) as RT60,
        anyLast(RT30) as RT30,
        anyLast(RT20) as RT20,
        anyLast(RT10) as RT10,
        anyLast(RT5) as RT5,
        

        sum(vol) as vol,
        sum(amt) as amt
        from {self.input_name}
        group by (date,toInt32(floor((time-0.5)/{self.period})))
        """
        return _sql

########################

if __name__ == "__main__":
    tk0 = table_kline_tick("rb")
    tk0.insert_data()

    tkp_dict = dict()
    for i in [5, 10, 15, 20, 30, 60, 90, 150, 300]:
        tkp_dict[i] = table_kline_period("rb", i)
        tkp_dict[i].insert_data()
    
    read_sql("""select * from rb.kline_150""")
