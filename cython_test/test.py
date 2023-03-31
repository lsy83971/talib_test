from importlib import reload
import tss
reload(tss)
import jump_span
import pandas as pd
import numpy as np
x = pd.DataFrame([[1, 2], [3, 4]]).values.astype(np.float32)

output = tss.test1(x)


type(output[0])

gg = pd.Series(range(100000))
from datetime import datetime
print(datetime.now())
gg.loc[[1, 2, 3]]
print(datetime.now())

print(datetime.now())
gg.loc[10001:11000]
print(datetime.now())

import cProfile
cProfile.run("gg.loc[10001:11000]")












