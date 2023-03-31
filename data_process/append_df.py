import pandas as pd

def cc1(df, f):
    df1 = pd.DataFrame(df.apply(f, axis=1).tolist())
    df = df.loc[:, (~df.columns.isin(df1.columns))]
    return pd.concat([df, df1], axis=1)

def cc2(df, f):
    df1 = pd.DataFrame(f(df))
    cols = list()
    for i in list(df.keys()).copy():
        if i not in df1.columns:
            cols.append(i)

    df_new = pd.concat([df[cols], df1], axis=1)
    # if hasattr(df, "attr"):
    #     df_new.attr = df.attr
    # else:
    #     df_new.attr = dict()
    return df_new


