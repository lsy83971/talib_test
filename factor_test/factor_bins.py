import pandas as pd
pd.set_option("display.max_columns", 30)
pd.set_option('display.float_format',  '{:.2f}'.format)
from data_process.corr_analyze import corr_pcluster
TXM_cluster = corr_pcluster("rb", corr_talib_m, y_symbol="^O{0,1}M{0,1}R[MT]")


idy = [
    "MRT10",     
    "MRT20", 
    "MRT30", 
    "MRT60", 
    "MRM1", 
    "MRM2", 
    "MRM3",
    "MRM5",    
]
i = "TXM_TRIMA"
txm1 = TXM_cluster[1]. data
txm = TXM_cluster[600]. data
txm. b2(x=i, y=idy)

txm[idy]. mean()

txm["OMRM1"]. mean()
txm["OMRM2"]. mean()
txm["OMRM3"]. mean()
txm["OMRM5"]. mean()

txm["close"]


txm1[["close", "time", "date"]]. sort_values(["date", "time"])
txm1[["close", "time", "date"]]

(txm1["close"]. shift( -120) - txm1["close"]).mean()
(txm1["close"]. shift( -120) - txm1["close"]).mean()
txm1["OMRM1"]. mean()


gg = txm1[["OMRM1", "time", "date"]]. merge(txm[["OMRM1", "time", "date"]], on=["date", "time"])

(gg["OMRM1_x"] - gg["OMRM1_y"]).abs().mean()

t_gp600 = txm1["time"]. apply(lambda x:x % 600)
t_gp300 = txm1["time"]. apply(lambda x:x % 300)
t_gp120 = txm1["time"]. apply(lambda x:x % 120)
t_gp60 = txm1["time"]. apply(lambda x:x % 60)
txm1["MRM1"].groupby(t_gp300).apply(lambda x:x.mean())
txm1["MRM1"].groupby(t_gp120).apply(lambda x:x.mean())
txm1["MRM1"].groupby(t_gp60).apply(lambda x:x.mean())

txm1["MRT5"].groupby(t_gp300).apply(lambda x:x.mean()).sort_values()
txm1["MRT5"].groupby(t_gp120).apply(lambda x:x.mean())
txm1["MRT5"].groupby(t_gp60).apply(lambda x:x.mean())




gg = txm1[["OMRM1", "time", "date"]]. merge(txm[["OMRM1", "time", "date"]], on=["date", "time"], how="right")
gg[gg.cc("OM")]. mean()


pd.set_option("display.max_rows", 1000)
pd.set_option('display.float_format',  '{:.3f}'.format)
