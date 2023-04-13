from local_sql import exch_detail, read_sql, client
from kline import table_kline_period, table_kline_tick, table_kline_period_whole, kline_period
from talib_idx import table_talib_normal, table_talib_period_whole

d1 = "20221105"
d2 = "20230404"
#d2 = "20221110"    
code = "rb"

def load_data(code, d1, d2):
    print("STEP1: Insert tickdata")
    tick_table = exch_detail(code)
    tick_table.insert_dates(d1, d2)
    
    print("STEP2: Insert kline")
    print("create table: kline_1")
    tk0 = table_kline_tick(code)
    tk0.insert_data()

    print("create table: kline_whole")
    tk0 = table_kline_period_whole(code)
    tk0.insert_data()
    
    tkp_dict = dict()
    for i in kline_period[1:]:
        print(f"create table: {code}.kline_{i}")
        tkp_dict[i] = table_kline_period(code, i)
        tkp_dict[i].insert_data()

    print("STEP3: Insert TXV(talib index basic)")
    for i in kline_period:
        print(f"create table: {code}.TXV_{i}")        
        TXV = table_talib_normal(code=code,
                                 close_idx="VWAP",
                                 surfix="TXV", 
                                 input_table=f"kline_{i}",
                                 output_table=f"TXV_{i}")
        TXV.insert_data()
        TXV = table_talib_period_whole(code="rb",
                                       close_idx="VWAP",
                                       period=i, 
                                       surfix="TXV",
                                       input_table="kline_whole",
                                       output_table=f"TXV_{i}_whole"
                                       )
        TXV.insert_data()
        

    print("STEP4: Insert TXM(talib index mid)")
    for i in kline_period:
        print(f"create table: {code}.TXM_{i}")        
        TXM = table_talib_normal(code=code,
                                 close_idx="MID",
                                 surfix="TXM", 
                                 input_table=f"kline_{i}",
                                 output_table=f"TXM_{i}")
        TXM.insert_data()
        TXM = table_talib_period_whole(code="rb",
                                       close_idx="MID",
                                       period=i, 
                                       surfix="TXM",
                                       input_table="kline_whole",
                                       output_table=f"TXM_{i}_whole"
                                       )
        TXM.insert_data()

        # df = read_sql(TXM._get_sql())
        # df
        
        
# cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

if __name__ == "__main__":
    d1 = "20221105"
    d2 = "20230404"
    #d2 = "20221110"    
    code = "rb"
    load_data(code, d1, d2)

    gg = read_sql("select * from rb.TXM_300 limit 100")
    gg.iloc[0]

    # gg[["ORT30", "OMRT30"]]. corr()
    # gg[["ORT5", "OMRT5"]]. corr()
    
    # client.execute("drop table rb.kline_0")
    # client.execute("drop table rb.kline_tick")
    # client.execute("drop table rb.TXV_0")            
    # client.execute("drop table rb.TXV_tick")        
    # df.iloc[0]
    #(df["close"] - df["high"] >= 0).mean()
    #(df["close"] - df["low"] < 0).mean()

    # d1 = "20221105"
    # d2 = "20230404"
    # code = "ru"
    # load_data(code, d1, d2)

    # d1 = "20221105"
    # d2 = "20230404"
    # code = "cu"
    # load_data(code, d1, d2)


    # gg = read_sql("""
    # select * from cu.kline_300 
    # """)
    
    # gg = read_sql("""
    # select a.RT5,b.WAP from rb.kline_tick as a
    # inner join rb.tickdata as b
    # on a.date=b.date
    # and a.time=b.time
    # """)
    
    gg = read_sql("select count(1) from rb.TXV_600_whole")
    # gg.iloc[0]
    # #read_sql("select RT10 from rb.kline_5"). abs().mean()
    
