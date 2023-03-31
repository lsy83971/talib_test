# -*- coding: utf-8 -*-  
from bins import *
import json


def normalize_dict(js):
    if isinstance(js, list):
        return [normalize_dict(i) for i in js]
    if isinstance(js, dict):
        return {i: normalize_dict(j) for i, j in js.items()}
    tp = str(type(js))
    if "numpy" in tp:
        if "int" in tp:
            return int(js)

        if "float" in tp:
            return float(js)
    return js


def find_type(obj):
    if isinstance(obj, filter_interval):
        return "filter_interval"

    if isinstance(obj, filter_category):
        return "filter_category"

    if isinstance(obj, bins_kappa_mono):
        return "bins_kappa_mono"

    if isinstance(obj, bins_kappa):
        return "bins_kappa"

    if isinstance(obj, bins_count):
        return "bins_count"
    raise


def to_dict(obj):
    res = dict()
    tp = find_type(obj)
    if tp in ["filter_interval", "filter_category"]:
        res["type"] = tp
        res["raw"] = obj.raw
        res["attr"] = obj.attr
        return res

    if tp in ["bins_kappa", "bins_kappa_mono", "bins_count"]:
        res["type"] = tp
        res["data"] = [to_dict(fi) for fi in obj]
        return res
    raise


def from_dict(res):
    tp = res["type"]
    if tp in ["filter_interval", "filter_category"]:
        obj = eval(tp)(**res["raw"])
        obj.attr = res["attr"]
        return obj

    if tp in ["bins_kappa", "bins_kappa_mono", "bins_count"]:
        tmp_list = [from_dict(tmp_res) for tmp_res in res["data"]]
        obj = eval(tp)(tmp_list)
        obj.dev_init()
        return obj
    raise


def to_json(obj):
    return json.dumps(normalize_dict(to_dict(obj)))


def from_json(res):
    return from_dict(json.loads(res))


def to_file(obj, file_name):
    js = to_json(obj)
    with open(file_name, "w") as f:
        f.write(js)


def from_file(file_name):
    with open(file_name, "r") as f:
        js = f.read()
    return from_json(js)
