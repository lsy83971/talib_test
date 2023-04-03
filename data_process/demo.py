import pandas as pd
pd.set_option("display.max_rows", 500)
import numpy as np
import math
from append_df import cc1, cc2
import sys
sys.path.append("/home/lishiyu/Project/bin_tools")
from bins import binning, bins_simple_mean
from pvdict_utils import *
from exch_detail import append_feature_exch_detail
from tick_detail import append_tick_detail
from get_feature_data import get_feature_data
from timeseries_detail import *
from clickhouse_driver import Client
from pandas.api.types import infer_dtype
from sql import rsq, read_sql, get_kline
import warnings
warnings.simplefilter('ignore')
import re
import gc
from timeseries_detail import func_info, total_func_info
from corr_analyze import xydata, beautify_excel
from jump_span import EMA as EMA1
from talib.abstract import * 

df = pd.read_pickle("./test_data/kline.pkl")



ADX(df1, 60)












