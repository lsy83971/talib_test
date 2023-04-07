import pandas as pd
import numpy as np
import math
#from importlib import reload
#import common.dict_operator
#reload(common.dict_operator)
from common.dict_operator import DPop
## 2.show pv-dict

def show_price_tab(x, Lag=""):
    idx = ["Ask5", "Ask4", "Ask3", "Ask2", "Ask1",
                             "Bid1", "Bid2", "Bid3", "Bid4", "Bid5",
                             ]
    x1 = pd.DataFrame(index=idx,
                      columns=["Price", "Volume"]
                      )
    for i in x1.index:
        for j in x1.columns:
            x1.loc[i, j] = x.loc[i[:3] + j + i[3:] + Lag]

    return x1


def show_price_tab_change(x):
    ptb = pd.concat([show_price_tab(x, "_Lag5").set_index("Price"), 
               show_price_tab(x, "_Lag4").set_index("Price"), 
               show_price_tab(x, "_Lag3").set_index("Price"), 
               show_price_tab(x, "_Lag2").set_index("Price"), 
               show_price_tab(x, "_Lag1").set_index("Price"), 
               show_price_tab(x, "").set_index("Price")    ], axis=1).sort_index(ascending=False).fillna("")
    ptb.columns = ["Lag5", "Lag4", "Lag3", "Lag2", "Lag1", "this"]
    ptb0 = pd.DataFrame("", columns=ptb.columns, index=[""])
    ptb1 = pd.DataFrame(columns=ptb.columns)
    ptb1.loc["A"] = x[["Turnover_DiffLen1S_Lag5",
                            "Turnover_DiffLen1S_Lag4",
                            "Turnover_DiffLen1S_Lag3",
                            "Turnover_DiffLen1S_Lag2",
                            "Turnover_DiffLen1S_Lag1",
                            "Turnover_DiffLen1S",                            
                            ]]. tolist()
    ptb1.loc["V"] = x[["Volume_DiffLen1_Lag5",
                            "Volume_DiffLen1_Lag4",
                            "Volume_DiffLen1_Lag3",
                            "Volume_DiffLen1_Lag2",
                            "Volume_DiffLen1_Lag1",
                            "Volume_DiffLen1",                            
                            ]]. tolist()
    ptb1.loc["A"] = ptb1.loc["A"] - ptb1.loc["V"] * x["BidPrice1"]
    print(x["BidPrice1"])
    return pd.concat([ptb, ptb0, ptb1])
