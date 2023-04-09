from local_sql import exch_detail, read_sql
from kline import table_kline_period, table_kline_tick
from talib_idx import table_talib_normal

def load_data(code, d1, d2):
    print("STEP1: Insert tickdata")
    tick_table = exch_detail(code)
    tick_table.insert_dates(d1, d2)
    
    print("STEP2: Insert kline")
    print("create table: kline_tick")
    tk0 = table_kline_tick(code)
    tk0.insert_data()
    tkp_dict = dict()
    
    for i in [5, 10, 15, 20, 30, 60, 90, 150, 300]:
        print(f"create table: {code}.kline_{i}")
        tkp_dict[i] = table_kline_period(code, i)
        tkp_dict[i].insert_data()

    print("STEP3: Insert TXB(talib index basic)")
    for i in ["tick", 5, 10, 15, 20, 30, 60, 90, 150, 300]:
        print(f"create table: {code}.TXB_{i}")        
        TXB = table_talib_normal(code, f"kline_{i}", f"TXB_{i}")
        
        TXB.insert_data()

if __name__ == "__main__":
    d1 = "20221105"
    d2 = "20230404"
    code = "rb"
    load_data(code, d1, d2)
    df = TXB.get_join_data()
    # df.iloc[0]
    #(df["close"] - df["high"] >= 0).mean()
    #(df["close"] - df["low"] < 0).mean()

    d1 = "20221105"
    d2 = "20230404"
    code = "ru"
    load_data(code, d1, d2)

    d1 = "20221105"
    d2 = "20230404"
    code = "cu"
    load_data(code, d1, d2)


    gg = read_sql("""
    select * from cu.kline_300 
    """)
    
    gg = read_sql("""
    select a.RT5,b.WAP from rb.kline_tick as a
    inner join rb.tickdata as b
    on a.date=b.date
    and a.time=b.time
    """)
    
    gg = read_sql("select * from rb.TXB_300")
    gg.iloc[0]
    #read_sql("select RT10 from rb.kline_5"). abs().mean()
    
