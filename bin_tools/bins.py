# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import stats
import math

def guess_type(l):
    try:
        l = pd.Series(l).astype(np.float)
        return "int", l
    except:
        return "str", l


def reset_name(self, name, inplace=False):
    """
    将index命名为name 作为新的一列赋值添加到dataFrame第一列
    """
    self.index.name = name
    return self.reset_index(inplace=inplace)
pd.DataFrame.reset_name = reset_name


def cc_df(self, w):
    return pd.Series(self.columns[self.columns.str.contains(w)])
def ncc_df(self, w):
    return pd.Series(self.columns[~self.columns.str.contains(w)])
pd.DataFrame.cc = cc_df
pd.DataFrame.ncc = ncc_df

def cc_series_value(self, w):
    return self[self.str.contains(w)]
def ncc_series_value(self, w):
    return self[~self.str.contains(w)]
pd.Series.cc = cc_series_value
pd.Series.ncc = ncc_series_value



class filter_fit:
    def fit_data(self, func, **kwargs):
        info = func(**kwargs)
        self.attr = info

    def fit(self, func, **kwargs):
        assert "x" in kwargs
        cond = self.fit_mask(kwargs["x"])
        kwargs1 = {i: j.loc[cond] if (isinstance(j, pd.Series) or isinstance(j, pd.DataFrame))
                   else j for i, j in kwargs.items()}
        self.fit_data(func, **kwargs1)


class filter_interval(filter_fit):
    def __init__(self, left, right, lt, rt):
        self.is_stable = False
        self.left = left
        self.right = right
        self.lt = lt
        self.rt = rt

        self.check_valid()

    @property
    def bin_name(self):
        lts = "[" if self.lt == "close" else "("
        rts = "]" if self.rt == "close" else ")"
        if not ((self.left == math.inf) or (self.left == -math.inf)):
            ls = str(round(self.left * 1000) / 1000)
        else:
            ls = str(self.left)

        if not ((self.right == math.inf) or (self.right == -math.inf)):
            rs = str(round(self.right * 1000) / 1000)
        else:
            rs = str(self.right)

        return lts + ls + ", " + rs + rts

    def check_valid(self):
        if self.left > self.right:
            raise BoundError("left bound larger than right bound")
        if self.left == self.right:
            if not ((self.lt == "close") and (self.rt == "close")):
                raise BoundError("equal bound not close")

    def fit_mask(self, x):
        if self.lt == "close":
            cond1 = (x >= self.left)
        else:
            cond1 = (x > self.left)
        if self.rt == "close":
            cond2 = (x <= self.right)
        else:
            cond2 = (x < self.right)
        return (cond1 & cond2)

    @property
    def raw(self):
        return {"left": self.left,
                "right": self.right,
                "lt": self.lt,
                "rt": self.rt,
                }


class filter_category(filter_fit):
    def __init__(self, bin_name):
        self.bin_name = bin_name
        self.check_valid()

    def check_valid(self):
        assert isinstance(self.bin_name, str)

    def fit_mask(self, x):
        return (x == self.bin_name)

    @property
    def raw(self):
        return dict()


class splt_cat:
    def __init__(self, x, cat_max=20):
        x = x.astype(str)
        x_cat = list(set(x))
        assert len(x_cat) <= cat_max
        fi_list = list()
        for i in x_cat:
            fi_list.append(filter_category(i))
        self.fi_list = fi_list


class splt_tick:
    def __init__(self, ticks, single_tick=True, stable_points=[]):
        stable_points = [float(i) for i in stable_points]
        ticks = pd.Series(ticks + stable_points).sort_values()
        if single_tick is False:
            ticks = [ticks.iloc[0]] + ticks.iloc[1:].drop_duplicates().tolist()
        else:
            ticks = ticks.drop_duplicates().tolist()
        if len(ticks) == 1:
            ticks = ticks * 2
        need_stablize = [i in stable_points for i in ticks]
        if single_tick is False:
            need_single_ticks = [i in stable_points for i in ticks]
        else:
            need_single_ticks = [True for i in ticks]

        fi_list = list()
        for i in range(len(ticks)):
            if i == 0:
                if need_single_ticks[i]:
                    fi = filter_interval(left=ticks[i], right=ticks[i],
                                         lt="close", rt="close")
                    fi_list.append(fi)
            else:
                lt = "open"
                if i == 1:
                    if need_single_ticks[0] == False:
                        lt = "close"

                if need_single_ticks[i] is False:
                    fi_list.append(filter_interval(left=ticks[i - 1], right=ticks[i],
                                                   lt=lt, rt="close"))
                else:
                    fi_list.append(filter_interval(left=ticks[i - 1], right=ticks[i],
                                                   lt=lt, rt="open"))
                    fi = filter_interval(left=ticks[i], right=ticks[i],
                                         lt="close", rt="close")
                    fi_list.append(fi)

            if need_stablize[i] is True:
                fi.is_stable = True

        self.fi_list = fi_list


class splt_eqfrq:
    def __init__(self, x, quant, single_tick=True, stable_points=[]):
        stable_points = [float(i) for i in stable_points]
        ticks = x.quantile([i / quant for i in range(quant + 1)], interpolation="nearest").tolist()
        self.fi_list = splt_tick(ticks=ticks, single_tick=single_tick, stable_points=stable_points).fi_list


class bins(list):
    def __init__(self, l):
        super().__init__(l)
        self.check_valid()
        self.back_link()

    def check_valid(self):
        pass

    def back_link(self):
        for i in self:
            i.up = self

    def fit_func(sekf, **kwargs):
        raise

    def fit_basic(self, **kwargs):
        for ft in self:
            ft.fit(func=self.fit_func, **kwargs)

    ## trans
    def trans(self, x, key=None):
        ts = pd.Series(index=x.index)
        for fi in self:
            ts.loc[fi.fit_mask(x).values] = fi.attr[key]
        return ts

    ## show
    def show(self):
        df = pd.DataFrame({fi.bin_name: fi.attr if hasattr(fi, "attr") else dict() for fi in self}).T
        return df.reset_name("bin_name")

    def show_all(self):
        df = pd.DataFrame({fi.bin_name: {**(fi.attr if hasattr(fi, "attr") else dict()), **fi.raw} for fi in self}).T
        return df.reset_name("bin_name")


class bins_interval(bins):
    ## init method
    ## fit_func, merge_func, dev_func_basic  需要定义

    def check_exact(self, i):
        i1 = self[i]
        i2 = self[i + 1]
        assert i1.right == i2.left

        if i1.rt == "close":
            assert i2.lt == "open"

        if i2.lt == "close":
            assert i1.rt == "open"

    def check_exact_total(self):
        for i in range(len(self) - 1):
            self.check_exact(i)

    def check_type(self, i):
        assert isinstance(self[i], filter_interval)
        self[i].check_valid()

    def check_type_total(self):
        for i in range(len(self)):
            self.check_type(i)

    def check_valid(self):
        self.check_type_total()
        self.check_exact_total()

    ## dev method

    def dev_func_basic(self, i):
        raise

    def dev_func(self, i):
        if self[i].is_stable or self[i + 1].is_stable:
            return math.inf
        else:
            return self.dev_func_basic(i)

    def dev_init(self):
        self.dev_list = [self.dev_func(i) for i in range(len(self) - 1)]

    def dev_update(self, i):
        self.dev_list[i] = self.dev_func(i)

    def dev_update_merge(self, i):
        self.dev_list.pop(i)
        if i > 0:
            self.dev_update(i - 1)
        if i < (len(self) - 1):
            self.dev_update(i)

    def dev_min(self):
        dl = pd.Series(self.dev_list)
        idx = dl.idxmin()
        value = dl.loc[idx]
        return {"idx": idx, "value": value}

    ## merge method

    def merge_basic(self, i):
        self.check_exact(i)
        left = self[i].left
        lt = self[i].lt
        right = self[i + 1].right
        rt = self[i + 1].rt
        fi = filter_interval(left=left, right=right, lt=lt, rt=rt)
        fi.up = self
        return fi

    def merge_func(self, i1, i2):
        return dict()

    def merge_fit(self, i):
        fi = self.merge_basic(i)
        fi1 = self[i]
        fi2 = self[i + 1]
        res = self.merge_func(fi1, fi2)
        fi.attr = res
        return fi

    def merge(self, i):
        self[i] = self.merge_fit(i)
        self.pop(i + 1)
        self.dev_update_merge(i)

    ## scis
    def scis(self, rule):
        self.dev_init()
        while True:
            if len(self) <= 1:
                break
            dev_min = self.dev_min()
            idx = dev_min["idx"]
            value = dev_min["value"]
            if value > rule:
                break

            else:
                self.merge(idx)

    def normalize(self):
        self[0].left = -math.inf
        self[- 1].right = math.inf


class bins_category(bins):
    pass


def fit_woe(self, x, y):
    cnt = x.shape[0]
    bad = y.sum()
    good = cnt - bad
    mean = bad / (cnt + 0.5)
    woe = math.log((bad + 0.5) / (good + 0.5)) - self.woe
    return {
        "cnt": cnt,
        "bad": bad,
        "good": good,
        "mean": mean,
        "woe": woe,
    }


class binscat_kappa(bins_category):
    fit_func = fit_woe

    def fit_basic(self, x, y):
        bad = y.sum() + 0.5
        good = y.shape[0] - bad + 0.5
        self.woe = math.log(bad / good)
        super().fit_basic(x=x, y=y)


class bins_count(bins_interval):
    def fit_func(self, x):
        return {"cnt": x.shape[0]}

    def dev_func_basic(self, i):
        cnt1 = self[i].attr["cnt"]
        cnt2 = self[i + 1].attr["cnt"]
        cnt3 = min(cnt1, cnt2) + min(max(cnt1, cnt2) / 100000, 1)
        return cnt3

    def merge_func(self, i1, i2):
        return {"cnt": i1.attr["cnt"] + i2.attr["cnt"]}


def dev_kappa(g1, b1, g2, b2):
    total_cnt = (g1 + g2 + b1 + b2)
    assert total_cnt > 0

    pf = min(0.5, total_cnt / 10000)

    g1 += pf
    g2 += pf
    b1 += pf
    b2 += pf

    good_rate = (g1 + g2) / total_cnt
    bad_rate = 1 - good_rate
    cat1_rate = (g1 + b1) / total_cnt
    cat2_rate = 1 - cat1_rate

    g1_est = total_cnt * good_rate * cat1_rate
    b1_est = total_cnt * bad_rate * cat1_rate
    g2_est = total_cnt * good_rate * cat2_rate
    b2_est = total_cnt * bad_rate * cat2_rate

    kappa_v = 0
    kappa_v += (g1_est - g1) ** 2 / g1_est
    kappa_v += (b1_est - b1) ** 2 / b1_est
    kappa_v += (g2_est - g2) ** 2 / g2_est
    kappa_v += (b2_est - b2) ** 2 / b2_est
    return stats.chi2.cdf(kappa_v, 1)



class bins_simple_mean(bins_interval):
    def fit_func(self, x, y):
        return {"cnt":x.shape[0],"mean":y.mean(),"sum":y.sum()}
    ## trans

    def dev_func_basic(self, i):
        return -1

    def merge_func(self, i1, i2):
        d1 = i1.attr
        d2 = i2.attr
        _cnt=d1.cnt+d2.cnt        
        _sum=d1.sum()+d2.sum()
        _mean=_sum/_cnt
        return {
            "cnt": _cnt,
            "sum": _sum,
            "mean": _mean,
        }

class bins_kappa(bins_interval):
    fit_func = fit_woe

    def fit_basic(self, x, y):
        bad = y.sum() + 0.5
        good = y.shape[0] - bad + 0.5
        self.woe = math.log(bad / good)
        super().fit_basic(x=x, y=y)

    ## trans

    def dev_func_basic(self, i):
        g1 = self[i].attr["good"]
        b1 = self[i].attr["bad"]
        g2 = self[i + 1].attr["good"]
        b2 = self[i + 1].attr["bad"]
        return dev_kappa(g1, b1, g2, b2)

    def merge_func(self, i1, i2):
        d1 = i1.attr
        d2 = i2.attr
        cnt = d1["cnt"] + d2["cnt"]
        bad = d1["bad"] + d2["bad"]
        good = d1["good"] + d2["good"]
        mean = bad / (cnt + 0.5)
        woe = math.log((bad + 0.5) / (good + 0.5)) - self.woe
        return {
            "cnt": cnt,
            "bad": bad,
            "good": good,
            "mean": mean,
            "woe": woe,
        }


class bins_kappa_mono(bins_kappa):
    def dev_func_basic(self, i):

        g1 = self[i].attr["good"]
        b1 = self[i].attr["bad"]
        g2 = self[i + 1].attr["good"]
        b2 = self[i + 1].attr["bad"]
        assert (g1 + b1) >= 10
        assert (g2 + b2) >= 10

        kv = dev_kappa(g1, b1, g2, b2)

        mono_way = self.mono_list[i]
        mono_cnt_left = 0
        mono_cnt_right = 0

        for mw in self.mono_list[:i][:: -1]:
            if mw == mono_way:
                mono_cnt_left += 1
            else:
                break

        for mw in self.mono_list[(1 + i):]:
            if mw == mono_way:
                mono_cnt_right += 1
            else:
                break

        mono_cnt = mono_cnt_left + mono_cnt_right
        mono_pvalue = (mono_cnt + 2) / (2 ** (mono_cnt + 1))

        kv1 = 1 - (1 - kv) * mono_pvalue
        return kv1

    def mono_init(self):
        self.mono_list = [self[i + 1].attr["mean"] > self[i].attr["mean"]
                          for i in range(len(self) - 1)]

    def dev_init(self):
        self.mono_init()
        super().dev_init()

    def dev_update_merge(self, i):
        self.dev_init()


def common_binning(
        x, y,
        ticks=None, quant=None,
        single_tick=True, stable_points=[],
        cnt_min=None,
        scis_class=bins_kappa,
        scis_dev_min=None,
):
    try:
        x = x.astype(float)
        x_type = "float"
    except:
        ef = splt_cat(x=x)
        x_type = "str"

    if x_type == "float":
        if ticks is not None:
            ef = splt_tick(ticks=ticks, single_tick=single_tick, stable_points=stable_points)
        else:
            assert quant is not None
            ef = splt_eqfrq(x=x, quant=quant, single_tick=single_tick, stable_points=stable_points)

        fi = bins_count(ef.fi_list)
        fi.fit_basic(x=x)
        if cnt_min is not None:
            fi.scis(cnt_min)

        if not scis_class is None:
            fi1 = scis_class(fi)
            fi1.fit_basic(x=x, y=y)
            if scis_dev_min is not None:
                fi1.scis(scis_dev_min)
        else:
            fi1 = fi

    else:
        fi1 = binscat_kappa(ef.fi_list)
        fi1.fit_basic(x=x, y=y)
    return fi1


def binning(self, x, y, **kwargs):
    fi1 = common_binning(x=self[x], y=self[y], **kwargs)
    return fi1.show()

def binning_simple(self, x, y, single_tick=False, quant=10, f=None, **kwargs):
    if f is None:
        if isinstance(y, list):
            f = lambda x, y:{"cnt": y.shape[0], **y.mean().to_dict()}
            #f = lambda x, y:{"cnt": y.shape[0], "cnt1": y.shape[1]}            
        else:
            f = lambda x, y:{"cnt": y.shape[0], "mean": y.mean()}
    

    class tmp_cls(bins_interval):
        def fit_func(self, x, y):
            return f(x, y)
        def dev_func_basic(self, i):
            return -1


    fi1 = common_binning(x=self[x], y=self[y],
                         single_tick=single_tick,
                         quant=quant, 
                         scis_class = tmp_cls, 
                         **kwargs)
    return fi1.show()


f_med = lambda x, y:{"cnt": y.shape[0], "mean": y.mean(), "median": y.median()}
    

pd.DataFrame.b1 = binning
pd.DataFrame.b2 = binning_simple


def r1(self, d):
    c = pd.Series(self.columns)
    self.columns = c.apply(lambda x:d.get(x, x))
pd.DataFrame.r1 = r1
