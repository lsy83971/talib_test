import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype
from clickhouse_driver import Client
import re
from data_process.remote_sql import get_daily_max
from common.append_df import cc2
from bin_tools import bins
import sys
import multiprocessing
from multiprocessing import Pool

client = Client(host='10.20.128.88', database='', user='default', password='irelia17')

def read_sql(sql):
    data, columns = client.execute(sql, columnar=True, with_column_types=True)
    df = pd.DataFrame({re.sub(r'\W', '_', col[0]): d for d, col in zip(data, columns)})
    df.colinfo = pd.DataFrame(columns, columns=["name", "typesql"])
    return df


