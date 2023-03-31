import requests
import pickle
import os
from collections import defaultdict
import pandas as pd
import numpy as np


def clickhouse_query(query):
    query=query+" format CSVWithNames"
    response = requests.post(
        'http://{}:8123?'.format('clickhouse.db.prod.highfortfunds.com'),
        data=query,
        auth=('readonly', ''),
        stream=True)
    df = pd.read_csv(response.raw)
    return df

clq=clickhouse_query


