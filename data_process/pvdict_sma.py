from data_process.kline import table_pipeline_sql
class table_pipeline_sql_Dcumsum(table_pipeline_sql):
    orderby = "date,time"
    cs_ls = [
        "D_ask_old",
        "D_ask_new",
        "D_ask_add",
        "D_ask_cancel",
        "D_ask_exch",        
        
        "D_bid_old",
        "D_bid_new",
        "D_bid_add",
        "D_bid_cancel",
        "D_bid_exch",

        "D_mid_exch", 
    ]
    period = [1, 5, 10, 20, 30, 60, 120, 240, 360, 600]
    def _insert_sql(self):
        sql = f"""
        select {
        " ".join([f"mapFilValueNozero(b.{i}_cumsum_{j}) as {i}_cumsum_{j}," for i in self.cs_ls for j in self.period])
        }
        b.date,
        b.Session,
        b.time
        from
          (select
            {
            " ".join(
            [f"sumMap(a.{i}_dif_{j}) over (PARTITION by a.date,a.Session order by a.time) as {i}_cumsum_{j}," for i in self.cs_ls for j in self.period]
            )
            }
            a.date,
            a.Session,
            a.time
            from 
            (select
            {
            " ".join(
            [f"mapSubtract({i},any({i}) over w{j}) as {i}_dif_{j}," for i in self.cs_ls for j in self.period]
            )
            }
            date,
            Session,
            time
            from {self.input_name}
            WINDOW
            {
            ",".join(
            [f"w{j} as (partition by date,Session order by date,time Rows BETWEEN {j} PRECEDING and {j} PRECEDING)" for j in self.period]
            )
            }
            ) as a
          ) as b
        

        """
        return sql

if __name__ == "__main__":
    tab = table_pipeline_sql_Dcumsum("rb", "tickdata", "DSMA")
    tab.drop()
    tab.insert_data()

