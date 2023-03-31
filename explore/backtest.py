import pandas as pd
from jump_span import jump

def max_drawdown(ts):
    ts_future_min = (ts[:: -1]. cummin())[:: -1]
    ts_drawdown = -(ts_future_min - ts)
    return ts_drawdown

class ret_ts:
    def __init__(self, y, fric=0):
        assert isinstance(y, pd.Series)
        self.y = y
        self.fric = fric
        
    def test(self, cond):
        cond = cond & (~self.y.isnull())
        self.Mean = cond.mean()
        self.Sum = cond.sum()
        y = self.y.copy()
        y.loc[~cond] = 0
        self.yRet = y
        
        yPlay = y.loc[cond]
        self.RMean = yPlay.mean()
        self.RSum = yPlay.sum()
        self.RStd = yPlay.std()

        self.WinRate = (yPlay > 0).mean()
        self.WinRMean = yPlay[yPlay > 0]. mean()
        self.LoseRate = (yPlay < 0).mean()
        self.LoseRMean = yPlay[yPlay < 0]. mean()

        self.RCumsum = self.yRet.cumsum()
        self.RDrawdown = max_drawdown(self.RCumsum)
        
        self.maxDrawdown = self.RDrawdown.max()
        self.calmarRatio = self.RSum / self.maxDrawdown
        self.sharpRatio = self.RMean / self.RStd

    def reform_cond(self, cond, span=60):
        cond = cond & (~self.y.isnull())
        l = jump(cond.values, 60)
        cond[:] = False
        cond.iloc[l] = True
        return cond

    def test1(self, cond, span=60):
        cond = self.reform_cond(cond, span=span)
        self.test(cond)
        
    def __repr__(self):
        return f"<{round(self.RMean,6)}>"

    def show_info(self, l=None):
        if l is None:
            l = ["Mean", "Sum", 
                 "RMean", "RSum", "RStd",
                 "WinRate", "WinRMean", "LoseRate", "LoseRMean",
                 #"retCumsum", "retDrawdown",
                 "maxDrawdown","calmarRatio", "sharpRatio"]
        return {i: getattr(self, i) for i in l}

class ret_tss(list):
    def show_info(self, l=None):
        return pd.DataFrame([i.show_info(l=l) for i in bt])
    def show_ts(self, key="retCumsum"):
        return pd.concat([getattr(i, key) for i in bt], axis=1)
        
def backtest(x, y, cond=None, quant=10, fric=0):
    if cond is not None:
        x = x.loc[cond]
        y = y.loc[cond]
    tick = sorted(list(set(x.quantile([i / quant for i in range(quant + 1)]).tolist())))
    bins_left = tick[: -1]
    bins_left[0] = -math.inf
    bins_right = tick[1:]
    bins_right[ - 1] = math.inf
    rts = ret_tss()
    rts.fric = fric
    for i in range(len(bins_left)):
        left_tick = bins_left[i]
        right_tick = bins_right[i]
        cond1 = (x > left_tick) & (x <= right_tick)
        y1 = y.copy()
        y1.loc[~cond1] = None
        rt = ret_ts(y1, fric=fric)
        rt.left_tick = left_tick
        rt.right_tick = right_tick
        rts.append(rt)
    return rts

if __name__ == "__main__":
    x = df1["bid_addord"]
    y = (1 + df1["ReturnLen20"] / df1["MidPriceLvl1"]).apply(lambda x:math.log(x))
    cond=None; quant=10
    
    bt = backtest(x, y, fric=0.00001)
    bt.show_info()
    bt.show_ts()



    

        
        
        


