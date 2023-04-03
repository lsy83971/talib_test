import pandas as pd
import numpy as np
from jump_span import jump

def max_drawdown(ts):
    ts_future_min = (ts[:: -1]. cummin())[:: -1]
    ts_drawdown = -(ts_future_min - ts)
    return ts_drawdown


def calc_Rinfo(yPlay):
    res = dict()
    RMean = yPlay.mean()
    RSum = yPlay.sum()
    RStd = yPlay.std()
    WinRate = (yPlay > 0).mean()
    WinRMean = yPlay[yPlay > 0]. mean()
    LoseRate = (yPlay < 0).mean()
    LoseRMean = yPlay[yPlay < 0]. mean()
    RDrawdown = max_drawdown(yPlay.cumsum())
    maxDrawdown = RDrawdown.max()
    calmarRatio = RSum / maxDrawdown
    sharpRatio = RMean / RStd
    return {
        "RMean":RMean, 
        "RSum":RSum, 
        "RStd":RStd, 
        "WinRate":WinRate, 
        "WinRMean":WinRMean, 
        "LoseRate":LoseRate, 
        "LoseRMean":LoseRMean, 
        #"RDrawdown":RDrawdown, 
        "maxDrawdown":maxDrawdown, 
        "calmarRatio":calmarRatio, 
        "sharpRatio":sharpRatio}


class RBT:
    def __init__(self, y, fric=0):
        assert isinstance(y, pd.Series)
        self.y = y
        self.fric = fric

    def test(self, cond):
        cond = cond.astype(np.int8)
        cond.loc[self.y.isnull()] = 0
        cond_play = cond != 0
        self.Mean = cond_play.mean()
        self.Sum = cond_play.sum()
        y = (self.y * (cond > 0) - self.y * (cond < 0))
        self.yRet = y
        self.yCS = y.cumsum()
        self.yPlay = y.loc[cond != 0]
        self.calc_info()

    def reform_cond(self, cond, span=60):
        cond = cond.astype(np.int8)
        cond.loc[self.y.isnull()] = 0
        l = jump(cond.values, span=span)
        cond1 = cond.copy()
        cond1[:] = 0
        cond1.iloc[l] = cond.iloc[l]
        return cond1

    def test1(self, cond, span=60):
        cond = self.reform_cond(cond, span=span)
        self.test(cond)
        
    def __repr__(self):
        if hasattr(self, "RMean"):
            return f"<{round(self.RMean,6)}>"
        else:
            return f"<RBT object>"

    def calc_info(self):
        self._info = calc_Rinfo(self.yPlay)

    @property
    def info(self):
        _info = pd.Series(self._info)
        _info.loc["ExchCnt"] = self.Sum
        _info.loc["ExchRatio"] = self.Mean
        return _info.round(2)
        
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



    

        
        
        


