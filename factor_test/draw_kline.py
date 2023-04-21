from data_process.local_sql import exch_detail, read_sql
from common.show_pv_tab import show_excel

tab = exch_detail("rb")
tab.get_columns()
df = read_sql(f"""select {','.join(tab.col_type["name"].tolist())} from rb.tickdata order by
date,time limit 100000
""")
show_excel(df, 3600, 3600, "test1.xlsx")





