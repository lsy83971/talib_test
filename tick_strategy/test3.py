import pickle
import datetime
import os
import requests
import scipy
import math
import time
import itertools
import json
import warnings
import glob

import numpy as np
import pandas as pd

import sys
from data_utils import *
from feature import *


trading_time_interval_dict = {
    'night': (3600 * -3, 3600 * -1),
    'day1': (3600 * 9, 3600 * 10 + 60 * 15),
    'day2': (3600 * 10 + 60 * 30, 3600 * 11 + 60 * 30),
    'day3': (3600 * 13 + 60 * 30, 3600 * 15)
}
volume_multiple = 10
tick_size_in_yuan = 1
symbol_class = 'rb'
exchange = 'SHFE'



shared_raw_data_dict = dict()

start_time = datetime.datetime.today() - datetime.timedelta(80)
end_date = (datetime.datetime.today() - datetime.timedelta(70)).strftime('%Y-%m-%d')

date = (start_time + datetime.timedelta(2)).strftime('%Y-%m-%d')
is_new = False
user = ""
password = ""
import pdb

max_symbol = get_max_symbol(date, symbol_class)
pdb.set_trace()

append_data(shared_raw_data_dict, date, max_symbol, trading_time_interval_dict, is_new, exchange, user, password, volume_multiple)

s1=json.load(open("2023-03-03.json"))
s2=pd.DataFrame(s1)

pd.Series(s2["v2311-C-7300"].loc["info"]["long_margin_rate"])
pd.Series(s2.iloc[:,1].loc["info"])


pd.DataFrame(s1["info"]["long_margin_rate"])


# SHFE 上海期货交易所
# DCE 大连商品交易所
# CZCE 郑州商品交易所

# Turnover 成交额
