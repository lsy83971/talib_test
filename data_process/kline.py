from data_process.local_sql import table_ch, has_table, has_date, read_sql, client
import pandas as pd
import numpy as np

kline_period = [1, 5, 10, 20, 30, 40, 60, 120, 180, 300, 600]

period_map = {
    "RM30":  30 * 120, 
    "RM20":  20 * 120, 
    "RM15":  15 * 120, 
    "RM10":  10 * 120, 
    "RM5":  5 * 120, 
    "RM3":  3 * 120, 
    "RM2":  2 * 120, 
    "RM1":  1 * 120, 
    "RT60":  60, 
    "RT30":  30, 
    "RT20":  20, 
    "RT10":  10, 
    "RT5":  5,
    "RT3":  3,    
    "RT1":  1,    
}

period_map_total = {**{i:(0, j) for i, j in period_map.items()},
                    **{"O" + i:(1, j) for i, j in period_map.items()},
                    **{"M" + i:(2, j) for i, j in period_map.items()},
                    **{"OM" + i:(3, j) for i, j in period_map.items()},                                        
                    }


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

class table_pipeline_sql(table_pipeline):
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

class table_kline_tick(table_pipeline_sql):
    def __init__(self, code):
        super().__init__(input_table="tickdata",
                         output_table="kline_1",
                         code=code
                         )
        
    def _insert_sql(self):
        l_str = list()
        for i, j in period_map. items():
            for window, surfix in [("w1", "O"), ("w2", "")]:
                for ret_idx, surfix1 in [("VWAP", ""), ("MID", "M")]:
                    l_str.append(f"""
                    (case when (nth_value({ret_idx},{j+1}) over {window})=0 then null
                    else nth_value({ret_idx},{j+1}) over {window} END)-{ret_idx} as {surfix}{surfix1}{i}
                    """)
                    
        _sql = f"""
        SELECT date,Session,time,vMult,Symbol,
        case when vol=0 then null
        else arrayMax(mapKeys(mapFilValuePos(D_exch))) end as high,
        case when vol=0 then null
        else arrayMin(mapKeys(mapFilValuePos(D_exch))) end as low,
        MID,
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

class table_kline_period(table_pipeline_sql):
    orderby = "date,start_time"      
    def __init__(self, code, period):
        assert isinstance(period, int)
        assert period >= 1
        assert period <= 600
        assert 1800 % period == 0

        self.period = period
        super().__init__(input_table="kline_1",
                         output_table="kline_" + str(period),
                         code=code
                         )
    
    def _insert_sql(self):
        ret_idx_cluster = sorted(period_map_total, key=lambda x:period_map_total[x])
        ret_idx_str = " ". join([f"anyLast({i}) as {i}," for i in ret_idx_cluster])
        _sql = f"""
        select
        date,
        any(Session) as Session,
        any(vMult) as vMult,
        any(Symbol) as Symbol,
        max(high) as high,
        min(low) as low,
        anyLast(WAP) as WAP,
        anyLast(VWAP) as VWAP,
        anyLast(MID) as MID,        

        any(t.time) as start_time,
        anyLast(t.time) as time,
        --anyLast(time) as end_time,
        --any(time) as start_time,

        {ret_idx_str}

        sum(vol) as vol,
        sum(amt) as amt
        from {self.input_name} as t
        group by (date,toInt32(floor((2*t.time-1)/{self.period})))
        """
        return _sql

class table_kline_period_whole(table_pipeline_sql):
    orderby = "date,time"      
    def __init__(self, code):
        self.code = code
        super().__init__(input_table="kline_1",
                         output_table="kline_whole",
                         code=code
                         )
    
    def _insert_sql(self):
        str_windows = """
        w{i} AS (PARTITION BY Symbol ORDER BY date,time asc Rows
        BETWEEN {i1} PRECEDING AND current row)
        """
        str_idx = """
        max(high) over w{i} as high_{i},
        min(low) over w{i} as low_{i},
        sum(vol) over w{i} as vol_{i},
        sum(amt) over w{i} as amt_{i},
        count() over w{i} as count_{i}
        """
        
        ret_idx_cluster = sorted(period_map_total, key=lambda x:period_map_total[x])
        ret_idx_str = ",". join([f"{i}" for i in ret_idx_cluster])
        _sql = f"""
        select
        date,
        time,
        Session,
        vMult,
        Symbol,
        WAP,
        VWAP,
        MID,
        high as high_1,
        low as low_1,
        vol as vol_1,
        amt as amt_1,        
        {",".join([str_idx.format(i=i) for i in kline_period[1:]])},
        {ret_idx_str}
        from {self.input_name} as t
        WINDOW
        {",".join([str_windows.format(i=i,i1=i-1) for i in kline_period[1:]])}

        """
        #print(_sql)
        return _sql

########################
#kline_period = [5, 10, 20, 30, 40, 60, 120, 180, 300, 600]
#pd.Series(period_map_total)
if __name__ == "__main__":
    tk0 = table_kline_period_whole("rb")
    tk0.insert_data()

    tkp_dict = dict()
    for i in [5, 10, 15, 20, 30, 60, 90, 150, 300]:
        tkp_dict[i] = table_kline_period("rb", i)
        tkp_dict[i].insert_data()

    gg = read_sql("""select * from rb.kline_whole limit 10000""")
    gg.iloc[2000]

