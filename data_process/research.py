import pandas as pd
import lightgbm as lgb
from get_feature_data import get_feature_data
from exch_detail import append_feature_exch_detail
from tick_detail import append_tick_detail

d1 = "20230225"
d2 = "20230314"
n = "rb"
rb_data = get_feature_data(d1, d2, n)

df_list = list()
for i, j in rb_data.items():
    for df in j.values():
        df = append_feature_exch_detail(df)
        df = append_tick_detail(df)
        df_list.append(df)
    print(i)


    
t_df = pd.concat(df_list).reset_index(drop=True)
t_df.to_pickle("t_df.pkl")
    
feature_type = df.dtypes.astype(str)
numeric_features = pd.Series(feature_type.ncc("object").index)
future_idx = numeric_features.cc("^F_|Ret")
avaliable_idx = numeric_features.ncc("^F_|Ret|Price|Count")

except_idx = ["last_unstable", 
              "index", 
              "last_unstable1", 
              "ExchTimeOffsetUs", 
              "Volume", 
              "Turnover",
              "LocalTimeStamp",
              "LocalTime", 
              "VolumeMultiple", 
              ]
avaliable_idx = avaliable_idx[~avaliable_idx.isin(except_idx)]
# avaliable_idx = [
#     "AskVolume1",
#     "BidVolume1",
#     "AskVolume2",
#     "BidVolume2",
#     "AskVolume3",
#     "BidVolume3",
#     "AskVolume4",
#     "BidVolume4",
#     "AskVolume5",
#     "BidVolume5",
# ]

## 1. stable deal with AskPrice1

stable_can_dealwithB1 = t_df["F_can_dealwith_B1"] & t_df["is_stable"]
moveup1 = ((t_df["BidPrice1_Diff1"] == 1) & \
           (t_df["AskPrice1_Diff1"] == 1) & \
           (t_df["AskBidPrice_Intv"] == 0))
moveup1_can_dealwithB1 = ((t_df["BidPrice1_Diff1"] == 1) & \
                          (t_df["AskPrice1_Diff1"] == 1) & \
                          (t_df["AskBidPrice_Intv"] == 0)) & t_df["F_can_dealwith_B1"]

#df1 = t_df.loc[can_dealwithB1]
condition = moveup1_can_dealwithB1
df1 = t_df.loc[condition]
y_idx = "ReturnLen20"
df1[y_idx]. mean()

x = t_df.loc[condition, avaliable_idx]
y = t_df.loc[condition, y_idx]


num1 = int(x.shape[0] * 0.6)
num2 = int(x.shape[0] * 0.8)
x_train = x.iloc[:num1]
x_valid = x.iloc[num1:num2]
x_test = x.iloc[num2:]
y_train = y.iloc[:num1]
y_valid = y.iloc[num1:num2]
y_test = y.iloc[num2:]
dt_train = lgb.Dataset(x_train, y_train)
dt_valid = lgb.Dataset(x_valid, y_valid)
dt_test = lgb.Dataset(x_test, y_test)

params = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 8,
    'learnnig_rage': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1,
    "n_estimators": 50,
    "l2": 100,
    "l1": 100, 
}
est = lgb.train(params,
                train_set=dt_train,
                valid_sets=(dt_train, dt_valid),
                early_stopping_rounds=30)

for i in avaliable_idx:
    print("******************")
    print(i)
    df1[i] = df1[i]. fillna( -999)
    print(df1.b2(i, "ReturnLen20"))

# ******************
# bid_change_volume
#               bin_name     cnt      mean
# 0  [-14096.0, -1276.0]  1730.0 -0.244644
# 1    (-1276.0, -932.0]  1732.0 -0.286458
# 2     (-932.0, -733.0]  1724.0 -0.291570
# 3     (-733.0, -587.0]  1734.0 -0.253762
# 4     (-587.0, -470.0]  1721.0 -0.304601
# 5     (-470.0, -370.0]  1733.0 -0.313873
# 6     (-370.0, -261.0]  1724.0 -0.298894
# 7     (-261.0, -137.0]  1729.0 -0.187500
# 8       (-137.0, 68.0]  1725.0 -0.095770
# 9       (68.0, 4087.0]  1728.0  0.036940
