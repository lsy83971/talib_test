from local_sql import exch_detail, read_sql
from kline import table_kline_period, table_kline_tick

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
        print(f"create table: kline_{i}")
        tkp_dict[i] = table_kline_period(code, i)
        tkp_dict[i].insert_data()

if __name__ == "__main__":
    d1 = "20221105"
    d2 = "20230404"
    code = "rb"
    load_data(code, d1, d2)

    d1 = "20221105"
    d2 = "20230404"
    code = "ru"
    load_data(code, d1, d2)

    #read_sql("select RT5 from rb.kline_5"). abs().mean()
    #read_sql("select RT10 from rb.kline_5"). abs().mean()
