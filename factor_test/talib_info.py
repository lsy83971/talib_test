import pandas as pd
import numpy as np
import math
import talib
from talib.abstract import *

TX_function = talib.get_functions()
total_func_info = pd.DataFrame({i: eval(i).info for i in TX_function}).T
total_func_info["output_num"] = total_func_info["output_names"]. apply(len)
cond1 = total_func_info["output_num"] == 1
cond2 = total_func_info["output_num"] > 1
total_func_info.loc[cond1, "output_names"] = total_func_info.loc[cond1, "name"]. apply(lambda x:[x])
total_func_info.loc[cond2, "output_names"] = total_func_info.loc[cond2]. apply(
    lambda x:[x["name"] + "_" + i for i in x["output_names"]], axis=1)

func_info = total_func_info.loc[total_func_info["group"]. ncc("Math|Price Transform|Statistic Functions").index]
func_info = func_info.loc[func_info["name"]. ncc("MAVP").index]
single_func = func_info[func_info["output_num"] == 1]
multi_func = func_info[func_info["output_num"] > 1]
