import warnings
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
pd.set_option("display.max_row", 500)


@property
def h1(self):
    return self.head(1).T


pd.DataFrame.h1 = h1


def r1(self, s):
    self.columns = pd.Series(self.columns).apply(lambda x: s.get(x, x))


pd.DataFrame.r1 = r1


def cc(self, s):
    return pd.Series(self.columns[self.columns.str.contains(s)])


pd.DataFrame.cc = cc


def ncc(self, s):
    return pd.Series(self.columns[~self.columns.str.contains(s)])


pd.DataFrame.ncc = ncc


def cc_s(self, s):
    return self[self.str.contains(s)]


def ncc_s(self, s):
    return self[~self.str.contains(s)]


def cc_si(self, s):
    return self[self.index.str.contains(s)]


def cv(self, s):
    return pd.Series({i: self.get(i, i) for i in s}, name=self.name)


pd.Series.cv = cv
pd.Series.cc = cc_s
pd.Series.ncc = ncc_s
pd.Series.cci = cc_si


## bf1 bf -- binning function -----------------------------

def bf1(word):
    return lambda x: {"cnt": x.shape[0], "rate": x[word].mean()}


bf = lambda y: {"cnt": y.shape[0], "mean": y.mean()}


def unstack(_df2, ind):
    _df2_rate = _df2.set_index(["code_0", "code_1", "bin_0", "bin_1"])[ind].unstack(["code_0", "bin_0"])
    _df2_rate.index = _df2_rate.index.get_level_values(1)
    _df2_rate.columns = _df2_rate.columns.get_level_values(1)
    _df2_rate.index.name = ind
    _df2_rate.columns.name = ind
    return _df2_rate


def excel_sheets(save_file, _tabs, sheet_name="table", mode=None, index=False):
    if mode is None:
        if os.path.exists(save_file):
            mode = "a"
        else:
            mode = "w"
    with pd.ExcelWriter(save_file, mode=mode, engine="openpyxl", if_sheet_exists="overlay") as writer:
        _row = 0
        for i, _tab in enumerate(_tabs):
            _tab.to_excel(writer, sheet_name=sheet_name, startrow=_row, startcol=0, index=index)
            _row += (2 + _tab.shape[0])


