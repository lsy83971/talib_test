# -*- coding: utf-8 -*-  
import os
import re
from common import *
from datetime import datetime
import xlsxwriter
from collections import OrderedDict
from cluster import hcl_mean
from importlib import reload
from bins import *
from ent import db_ent
import copy
import draw
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from logit_train import *


def cond_part(dt, l):
    if isinstance(l, float):
        l = [l]
    assert isinstance(l, list)
    assert len(l) >= 1
    cond = list()
    dts = dt.quantile(l).tolist()
    cond.append(dt <= dts[0])
    for i in range(len(l) - 1):
        cond.append((dt <= dts[i + 1]) & (dt > dts[i]))
    cond.append(dt > dts[- 1])
    return [c.values for c in cond]


class bins_dict(OrderedDict):
    def fit(self, x, y):
        for i, j in self.items():
            j.fit_basic(x=x[i], y=y)

    def to_df(self):
        res_list = list()
        for i, j in self.items():
            j1 = j.show()
            j1.insert(0, "index", i)
            res_list.append(j1)
        return pd.concat(res_list).reset_index(drop=True)

    def ent(self, i):
        return db_ent(self[i].show()[["bad", "good"]].values)

    @property
    def entL(self):
        return pd.Series({i: self.ent(i) for i in self.keys()}, name="ent").sort_values(ascending=False)


class lgt:
    def __init__(self, x, y, cmt=None, need_guess_type=True,
                 na=".replace('', -999999).fillna(-1)"
                 ):
        self.raw_x = x
        self.raw_y = y
        assert x.shape[0] == y.shape[0]
        assert isinstance(x, pd.DataFrame)
        assert isinstance(y, pd.Series)
        self.x = eval("x.reset_index(drop = True)" + na)

        if need_guess_type:
            dtypes = self.x.dtypes.astype(str)
            trans_type_dict = dict()
            for i in dtypes.index:
                t = dtypes[i]
                if t != "object":
                    continue
                i1, i2 = guess_type(self.x[i])
                if i1 == "int":
                    print("as float", i)
                    trans_type_dict[i] = i2

            if len(trans_type_dict) > 0:
                _x2 = pd.DataFrame(trans_type_dict)
                self.x.drop(list(trans_type_dict.keys()), axis=1, inplace=True)
                self.x = pd.concat([self.x, _x2], axis=1)

        self.y = y.reset_index(drop=True)
        self.y.name = "label"

        if cmt is None:
            cmt = pd.Series(x.columns, index=x.columns, name="comment")
        else:
            cmt = cmt.cv(self.x.columns)
        self.cmt = cmt
        self.cmt.name = "注释"
        self.init_container()
        self.init_tsp_path()

    def init_container(self):
        if not hasattr(self, "bins"):
            self.bins = bins_dict()

        if not hasattr(self, "errors"):
            self.errors = set()

    def init_tsp_path(self):
        if not os.path.exists("./model_profile"):
            os.mkdir("./model_profile")
        self.tsp = "Lgt" + datetime.now().strftime("%Y%m%d_%H%M")
        self.tsp_path = "./model_profile/" + self.tsp

    def init_png_dir(self):
        if not hasattr(self, "png_dir"):
            self.png_dir = "./" + self.tsp_path + "_png/"
        if not os.path.exists(self.png_dir):
            os.mkdir(self.png_dir)
        self.binning_excel_path = './' + self.tsp_path + ".xlsx"
        self.png_dict = dict()

    def binning(self,
                cols=None,
                bin_sample=None,

                quant=15,
                single_tick=True,
                stable_points=[],
                cnt_min=10,

                scis_class=bins_kappa_mono,
                scis_dev_min=0.999,

                pass_error=True,

                ):

        """
        bin_sample分箱用样本
        """
        self.init_container()

        if bin_sample is None:
            y = self.y
        else:
            y = self.y.loc[bin_sample]

        self.woe = math.log(((y == 1).sum() + 0.5) / ((y == 0).sum() + 0.5))

        if cols is None:
            cols = self.x.columns.tolist()

        for i in cols:
            if bin_sample is None:
                x = self.x[i]
            else:
                x = self.x[i].loc[bin_sample]
            try:
                fi1 = common_binning(
                    x=x, y=y,
                    quant=quant, stable_points=stable_points, single_tick=single_tick,
                    cnt_min=cnt_min,
                    scis_class=scis_class, scis_dev_min=scis_dev_min
                )
                self.bins[i] = fi1

            except Exception as e:
                if pass_error:
                    self.errors.add(i)
                    continue
                else:
                    raise e

        bin_cnt = self.binning_cnt()
        drop_cols = bin_cnt[bin_cnt == 1].index.tolist()
        for i in drop_cols:
            del self.bins[i]
        self.mono_cols = drop_cols
        for i in self.bins.values():
            try:
                i.normalize()
            except:
                pass

    def binning_cnt(self):
        return pd.Series({i: len(j) for i, j in self.bins.items()})

    def sub_binning(self, conds, labels, cols=None):
        assert len(labels) == len(conds)
        self.sub_bins = dict()
        for i in range(len(conds)):
            tmp_bins = copy.deepcopy(self.bins)
            tmp_bins.fit(self.x.loc[conds[i]], self.y.loc[conds[i]])
            self.sub_bins[labels[i]] = tmp_bins

        self.dev_porp()
        self.dev_trend()

    def sub_binning_plot(self, cols=None, draw_bad_rate=True, upper_lim=None):
        self.init_png_dir()
        self.plot_path = dict()

        if cols is None:
            cols = self.bins.keys()
        for en_name in cols:
            cn_name = self.cmt.get(en_name, "")
            path = self.png_dir + re.sub("[/()（）<>]", "_", en_name + "##" + cn_name + ".png")
            self.plot_path[en_name] = path
            en_name_abbr = en_name[:35] + ("..." if len(en_name) > 35 else "")
            cn_name_abbr = cn_name[:30] + ("..." if len(cn_name) > 30 else "")

            title = en_name_abbr + "\n" + cn_name_abbr
            self.barplot(en_name, title=title, path=path, draw_bad_rate=draw_bad_rate, upper_lim=upper_lim)

    def barplot_df(self, i):
        dfs = list()
        for j0, (j1, j2) in enumerate(self.sub_bins.items()):
            df = j2[i].show()
            df["label"] = j1
            df["part"] = str(j0 + 1)
            df["porp"] = df["cnt"] / df["cnt"].sum()
            df.r1({"mean": "bad_rate", "bin_name": "bin"})
            df.reset_name("code", inplace=True)
            dfs.append(df)
        t_df = pd.concat(dfs)
        return t_df

    def barplot(self, i, title, path, draw_bad_rate=True, upper_lim=None):

        t_df = self.barplot_df(i)
        if upper_lim is not None:
            t_df["bad_rate"] = t_df["bad_rate"].apply(lambda x: min(x, upper_lim))
        draw.draw_bar(t_df,
                      title=title,
                           save=path,
                           draw_bad_rate=draw_bad_rate)

    def draw_excel_addpng(self, cols, worksheet):

        print(len(cols))
        for i in range(len(cols)):
            _i1, _i2 = int(i / 3), i % 3
            worksheet.write(2 * _i1, 2 * _i2 + 1, "{0}".format(cols[i]))
            draw.insert_image(worksheet,
                                   self.plot_path[cols[i]],
                                   row=2 * _i1 + 1,
                                   col=2 * _i2 + 1,
                                   x_scale=3,
                                   y_scale=2)

    def draw_binning_excel_addpng(self, path, cols=None, sheet_name="分箱图"):
        if cols is None:
            cols = list(self.plot_path.keys())
        cols = cols.copy()
        ent = self.bins.entL
        cols.sort(key=lambda x: ent.get(x.split("##")[0], 0), reverse=True)
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet(sheet_name)
        self.draw_excel_addpng(cols, worksheet)
        workbook.close()

    def draw_binning_excel(self, cols=None):
        self.draw_binning_excel_addpng(path=self.binning_excel_path, cols=cols)
        with pd.ExcelWriter(self.binning_excel_path, mode="a", engine="openpyxl") as writer:
            self.bins.to_df().to_excel(writer, sheet_name="全量", index=False)
            for i, j in self.sub_bins.items():
                j.to_df().to_excel(writer, sheet_name=i, index=False)

    def woe_update(self, cols=None):
        if not hasattr(self, "woevalue"):
            self.woevalue = pd.DataFrame(index=self.x.index)

        if cols is None:
            cols = self.bins.keys()

        for i in cols:
            self.woevalue[i] = self.bins[i].trans(self.x[i], "woe").astype(float)

    def corr_update(self, cols=None):
        if cols is None:
            df = self.woevalue
        else:
            df = self.woevalue[cols]
        self.corr = df.corr()

    def dev_trend(self):
        self.tsi = pd.Series()
        for k in self.bins.keys():
            tmp_res = [j[k].trans(self.x[k], "woe") for i, j in self.sub_bins.items()]
            tmp_res = pd.concat(tmp_res, axis=1)
            self.tsi.loc[k] = tmp_res.std(axis=1).mean()
        return self.tsi

    def dev_porp(self):
        self.psi_detail = dict()
        self.psi = pd.Series()
        for k in self.bins.keys():
            tmp_res = [j[k].show()["cnt"] for i, j in self.sub_bins.items()]
            tmp_res = pd.concat(tmp_res, axis=1) + 0.5
            tmp_res = tmp_res / tmp_res.sum()

            t_porp = self.bins[k].show()["cnt"] + 0.5
            t_porp = t_porp / t_porp.sum()

            mi_dev = tmp_res.T - t_porp
            ln_dev = (tmp_res.T / t_porp).applymap(lambda x: math.log(x))

            self.psi_detail[k] = (mi_dev * ln_dev).T
            self.psi[k] = self.psi_detail[k].sum().max()

        return self.psi

    def cluster(self, cols=None, n=2, min_value=0.2):
        corr = self.corr
        if cols is not None:
            corr = corr.loc[cols, cols]
        ch = hcl_mean(distance=(1 - corr ** 2))
        self.hcl = ch
        ch.iter(n=n, min_value=min_value)
        d = list(copy.deepcopy(self.hcl.gp).values())
        self.single_cols=tuple([i[0] for i in d if len(i) == 1])
        self.gp_cols=[i for i in d if len(i)>1]

    def cluster_draw(self):
        path = self.tsp_path + "_cluster.xlsx"
        ent = self.bins.entL
        workbook = xlsxwriter.Workbook(path)

        gp_cols=copy.deepcopy(self.gp_cols)
        gp_cols.sort(key=lambda x: len(x), reverse=True)
        for i, j in enumerate(gp_cols):
            cols = list(j)
            cols.sort(key=lambda x: ent.get(x), reverse=True)
            worksheet = workbook.add_worksheet(str(i + 1))
            self.draw_excel_addpng(cols=cols, worksheet=worksheet)
        worksheet = workbook.add_worksheet("single")
        self.draw_excel_addpng(cols=self.single_cols, worksheet=worksheet)
        workbook.close()

    def cond_result(self, cond, score_ticks=None, info=None):
        if score_ticks is None:
            score_ticks = self.score_ticks
        res = dict()
        y = self.y.loc[cond]
        x = self.woevalue.loc[cond, self.cols]
        score = pd.Series(self.model.predict_proba(x)[:, 1], index=x.index)

        res["KS"] = KS(y, score)
        res["AUC"] = roc_auc_score(y, score)

        z = pd.concat([y, score], axis=1)
        z.columns = ["label", "x"]
        b = z.b1(x="x", y="label", ticks=score_ticks, single_tick=False)
        res["binning"] = b
        return res

    def train(self,
              cols,
              train_cond,
              valid_conds=[],
              test_conds=[],
              train_class=lt3_ang,
              init_param={"penalty": "l2", "C": 0.5, "quant": 10},
              train_param={"ang": 1.5},
              quant=10,
              ):
        """
        train_class: 逐步回归细类
        init_param: 逐步回归初始化参数
        train_param: 逐步回归新增指标时 选指标参数
        """
        lt = train_class(x=self.woevalue[cols],
                         y=self.y,
                         train_cond=train_cond,
                         valid_conds=valid_conds,
                         test_conds=test_conds,
                         **init_param)
        self.lt = lt
        lt.recursive_train(**train_param)
        self.cond_d = OrderedDict()
        self.cond_d["train"] = train_cond
        for i, j in enumerate(valid_conds):
            self.cond_d["valid_" + str(i + 1)] = j
        for i, j in enumerate(test_conds):
            self.cond_d["test_" + str(i + 1)] = j

        self.obj_d = OrderedDict()
        self.obj_d["train"] = self.lt
        for i, j in enumerate(self.lt.valid):
            self.obj_d["valid_" + str(i + 1)] = j
        for i, j in enumerate(self.lt.test):
            self.obj_d["test_" + str(i + 1)] = j

        ### TODO
        ### model_result
        self.model_result = list()
        for k in range(len(self.lt.result)):
            tmp_d = OrderedDict()
            for i, j in self.obj_d.items():
                tmp_d[i] = self.add_info(j.result[k]["binning"].reset_name(i))
            self.model_result.append(tmp_d)

        ### iter-KS AUC
        self.iter_KS = pd.DataFrame()
        self.iter_AUC = pd.DataFrame()
        for i, j in self.obj_d.items():
            self.iter_KS[i + "_KS"] = [k["KS"] for k in j.result]
            self.iter_AUC[i + "_AUC"] = [k["AUC"] for k in j.result]
        self.iter_info = pd.concat([self.iter_KS, self.iter_AUC], axis=1)

        ### iter-binning
        self.iter_binning = list()
        for k in range(1, len(self.lt.result)):
            tmp_s = list()
            for i, j in self.obj_d.items():
                tmp_v = self.add_info(j.result[k]["binning"].reset_name(i + "_" + str(k)))
                tmp_s.append(tmp_v)
            self.iter_binning.append(pd.concat(tmp_s, axis=1))
        self.iter_coef = pd.DataFrame([i["coef"] for i in self.lt.result[1:]])
        self.select_iter()

    def add_info(self, df, i=None, avg=None, cnt_sum=None):
        if i is not None:
            df["index"] = i
        if ("index" in df.columns) & ("cn_name" not in df.columns):
            df["cn_name"] = df["index"].apply(lambda x: self.cmt.get(x, x))
        if ("cnt" in df.columns) & ("porp" not in df.columns):
            if cnt_sum is None:
                cnt_sum = df["cnt"].sum()
            df["porp"] = df["cnt"] / cnt_sum
        if ("mean" in df.columns) & ("lift" not in df.columns):
            if avg is None:
                avg = (df["mean"] * df["cnt"]).sum() / df["cnt"].sum()
            df["lift"] = df["mean"] / avg
        df.r1({"cnt": "总数", "porp": "占比", "lift": "提升", "mean": "坏率",
               "index": "指标名", "cn_name": "指标含义", "bin_name": "分箱区间"
               })
        return df

    def select_iter(self, num=-1):
        self.num = num
        self.model_info = self.lt.result[num]
        self.cols = self.model_info["cols"]
        self.model = self.model_info["model"]
        self.score_ticks = self.model_info["st"]
        self.coef = self.model_info["coef"]

    def online(self, cn_name, en_name, container="model_image1"):
        """
        kw["cn_name"] = "自营天衍模型V1"
        kw["en_name"] = "ZY_TY_V1"
        kw["container"] = "model_image1"
        """
        self.save()
        kw = dict()
        kw["model_name"] = cn_name
        kw["channel"] = en_name
        kw["container"] = container
        kw["date"] = datetime.now().strftime("%Y%m%d")
        kw["model_cols"] = "\n".join([str(i + 1) + ". " + j for i, j in enumerate(self.cols)])
        kw["model_cols_json"] = self.raw_x.iloc[0][self.cols].to_dict()

        os.system("cp {0} {1}.csv".format(self.model_sample_path, kw["channel"]))
        os.system("cp {0} {1}.pkl".format(self.devfile_path, kw["channel"]))
        os.system("cp {0} {1}模型报告.xlsx".format(self.model_report_path, kw["channel"]))
        if os.path.exists("/home/bozb/notebook/lsy/PYLIB/MODEL/FILE/online_file.txt"):
            with open("/home/bozb/notebook/lsy/PYLIB/MODEL/FILE/online_file.txt", "r") as f:
                fs = f.read()
            with open("{0}_{1}.txt".format(kw["model_name"], kw["channel"]), "w") as f:
                f.write(fs.format(**kw))

    def save(self):
        self.save_model_report()
        self.save_model_sample()
        self.save_devfile()

    def save_model_sample(self):
        path = self.tsp_path + "_sample.csv"
        self.model_sample_path = path
        cols = self.cols
        score = self.model.predict_proba(self.woevalue[cols])[:, 1]
        x = self.raw_x[cols]
        x["score"] = score
        x.sample(min(1000, self.raw_x.shape[0])).to_csv(path, index=False)

    def save_model_report(self):
        path = self.tsp_path + "_report.xlsx"
        if os.path.exists(path):
            os.remove(path)
        self.model_report_path = path
        res = dict()

        sp_info1 = pd.DataFrame({i: {"总数": j.y.shape[0], "坏样本数": j.y.sum(),
                                     "坏率": j.y.mean(),
                                     } for i, j in self.obj_d.items()}).T.reset_name("样本")
        sp_info2 = pd.DataFrame({i: {"KS": j.result[self.num]["KS"],
                                     "AUC": j.result[self.num]["AUC"],
                                     } for i, j in self.obj_d.items()}).T.reset_name("样本")
        res["1.1模型汇总"] = [sp_info1, sp_info2]

        res["2.1单指标分箱"] = [self.add_info(self.bins.to_df(),
                                         cnt_sum=self.y.shape[0],
                                         avg=self.y.mean()
                                         )]
        if hasattr(self, "png_dir"):
            self.draw_binning_excel_addpng(path=path, cols=self.cols, sheet_name="2.2单指标分箱图")
        res["3.1模型参数"] = [self.add_info(pd.DataFrame(self.coef, columns=["系数"]).reset_name("index"))]
        iter_num = pd.DataFrame([["迭代次数", str(int(self.num))]], columns=["要素", "取值"])
        res["3.2迭代汇总"] = [iter_num, self.iter_info.reset_name("iter")]
        res["3.3迭代分箱"] = self.iter_binning
        res["3.4迭代系数"] = [self.iter_coef.fillna(0)]

        ## 以下需要样本分箱
        res["4.1样本分箱"] = list(self.model_result[self.num].values())
        for i in ["month", "channel"]:
            if i not in self.x.columns:
                continue
            if i == "month":
                i1 = "4.2月份分箱"
            if i == "channel":
                i1 = "5.1渠道分箱"
            tmp_l = list()
            v = list(set(self.x[i]))
            v.sort()
            for k in v:
                cond = (self.x[i] == k)
                b = self.cond_result(cond, score_ticks=self.score_ticks)["binning"]
                tmp_l.append(self.add_info(b.reset_name(k)))
            res[i1] = tmp_l

        from common import excel_sheets
        for i in sorted(res.keys()):
            excel_sheets(path, res[i], i)

    def save_devfile(self):
        path = self.tsp_path + "_devfile.pkl"
        self.devfile_path = path
        res = dict()
        res["cols"] = self.cols
        res["model"] = self.model
        res["standard_woe"] = 0
        res["trans"] = dict()
        for i in res["cols"]:
            tmp_d = dict()
            tmp_d["mode"] = 'intb'
            tmp_d["name"] = i
            tmp_d["info"] = self.bins[i].show_all()
            res["trans"][i] = tmp_d

        with open(path, "wb") as f:
            pickle.dump(res, f)
