#  Aothor: Ethan Lee
#  E-mail: ethan2lee.coder@gmail.com

import pandas as pd

# 从excel读取机位编号与机位区域信息
data = pd.read_excel("./data/aircraft_position.xlsx", index_col=None)
a_p = data[["location", "map_x", "map_y"]]
a_p.dropna(axis=0, how='any', inplace=True)
a_p.set_index("location")[["map_x", "map_y"]].astype("int", copy=True)

## location:机场编号，APRON: 机位所在的区域
apron_region = data[['location', 'APRON']]
