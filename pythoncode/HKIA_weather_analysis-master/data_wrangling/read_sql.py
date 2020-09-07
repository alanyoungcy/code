#  Created at ${DATE}
#  Aothor: Ethan Lee
#  E-mail: ethan2lee.coder@gmail.com

import pandas as pd
from sqlalchemy import create_engine
from data_wrangling.aircraft_position import apron_region
import numpy as np

engine = create_engine('mysql+pymysql://hkia_v1:Asdf168!!@192.168.10.170:3306/HKIA_v1')


def percentage(series):
    """
    对列进行求平均
    :param series:
    :return:列的平均值，百分制形式表示
    """
    sum = np.sum(series)
    return series / sum * 100

arr_dep = ['A', 'D']  # 航班起飞状态
run_ways = ['07L', '07R', '25L', '25R']  # 航班被安排的跑道为主
weathers = ["AODB_MAJOR", "Strong_Wind", "Thunder_Storm", "abnormal_time", "normal_time"]  # 航班所处天气状态

# ### 以天气状态为维度，查看原停机位置分布
#
# FIRST_STAND_NUMBER = {}
# for weather in weathers:
#     sql = """
#   select FIRST_STAND_NUMBER,count(*)  as {} from {}
#   where (FIRST_STAND_NUMBER is not NULL AND  FIRST_STAND_NUMBER <> " ")
#   group by FIRST_STAND_NUMBER
#   ORDER BY COUNT(*) DESC
#     """.format(weather, weather)
#     FIRST_STAND_NUMBER[weather] = pd.read_sql(sql, engine)
#
# ### 以天气状态为维度，查看最终停机位分布
# LAST_STAND_NUMBER = {}
# for weather in weathers:
#     sql = """
#
#
#   select LAST_STAND_NUMBER,count(*)  as {} from {}
#   where (LAST_STAND_NUMBER is not NULL AND  LAST_STAND_NUMBER <> " ")
#   group by LAST_STAND_NUMBER
#   ORDER BY COUNT(*) DESC
#     """.format(weather, weather)
#     df = pd.read_sql(sql, engine)
#     ### 对数据进行简单的聚类分析
#     data = df[weather].values.reshape(-1, 1)
#     # 假如我要构造一个聚类数为3的聚类器
#     estimator = KMeans(n_clusters=4)  # 构造聚类器
#     estimator.fit(data)  # 聚类
#     label_pred = estimator.labels_  # 获取聚类标签
#     centroids = estimator.cluster_centers_  # 获取聚类中心
#     inertia = estimator.inertia_  # 获取聚类准则的总和
#     df['CATEGORY'] = label_pred
#     df['RATIO'] = round((df[weather] / df[weather].sum()) * 100, 3)
#     LAST_STAND_NUMBER[weather] = df
#
#
#
# ### 以跑道为维度，统计原机位安排的分布状况
# FIRST_STAND_NUMBER_RUN_WAYS = {}
# for run_way in run_ways:
#     for weather in weathers:
#         sql = """
#         select FIRST_STAND_NUMBER,count(*)  as {1}_{0} from {0}
#         where (FIRST_STAND_NUMBER is not NULL AND  FIRST_STAND_NUMBER <> " ")
#         and  R_RUNWAY = \"{1}\"
#         group by FIRST_STAND_NUMBER
#         ORDER BY COUNT(*) DESC;
#           """.format(weather, run_way)
#         name = "{}_{}".format(run_way, weather)
#         FIRST_STAND_NUMBER_RUN_WAYS[name] = pd.read_sql(sql, engine)
#
# ### 将运行的结果保存至excel
# with pd.ExcelWriter('./data/飞机跑道与停机坪位置安排.xlsx') as writer:
#     for key ,values in FIRST_STAND_NUMBER_RUN_WAYS.items():
#         values['RATIO'] = values[key] / values[key].sum() * 100
#         data=values.merge(apron_region,how='left',left_on="FIRST_STAND_NUMBER",right_on="location")
#         data.drop(axis=1, columns=['location'], inplace=True)
#         data.to_excel(writer, sheet_name=key)
#
#
# ### 以起飞或降落为维度，统计原机位安排的分布状况
# FIRST_STAND_NUMBER_ARR_DEP = {}
# for state in arr_dep:
#     for weather in weathers:
#         sql = """
#       select FIRST_STAND_NUMBER,count(*)  as {1}_{0} from {0}
#       where (FIRST_STAND_NUMBER is not NULL AND  FIRST_STAND_NUMBER <> " ")
#       and  ARR_DEP = \"{1}\"
#       group by FIRST_STAND_NUMBER
#       ORDER BY COUNT(*) DESC;
#         """.format(weather,state)
#         name = "{}_{}".format(state,weather)
#         FIRST_STAND_NUMBER_ARR_DEP[name] = pd.read_sql(sql, engine)
#
# ### 保存数据
# with pd.ExcelWriter('飞机起飞降落与停机坪位置安排.xlsx') as writer:
#     for key ,values in FIRST_STAND_NUMBER_ARR_DEP.items():
#         values['RATIO'] = values[key] / values[key].sum() * 100
#         data=values.merge(apron_region,how='left',left_on="FIRST_STAND_NUMBER",right_on="location")
#         data.drop(axis=1, columns=['location'], inplace=True)
#         data.to_excel(writer, sheet_name=key)
#
#
# ### 以航班为维度分析，统计原始机位安排分布状况
# FIRST_STAND_NUMBER_FLIGHT_ID = {}
# def ratio(df, column="AODB_MAJOR"):
#     df["RATIO"] = round(df[column] / df[column].sum() * 100, 1)
#     return df
#
# for weather in weathers:
#     sql = """
#     select FLIGHT_ID,FIRST_STAND_NUMBER,count(*)  as {0} from {0}
#     where (FIRST_STAND_NUMBER is not NULL AND  FIRST_STAND_NUMBER <> " ")
#     group by FIRST_STAND_NUMBER,FLIGHT_ID
#     ORDER BY COUNT(*) DESC ;
#     """.format(weather)
#     data = pd.read_sql(sql, engine)
#     data = data.groupby(["FLIGHT_ID"]).apply(ratio, column=weather)
#     data = data.merge(apron_region, how='left', left_on="FIRST_STAND_NUMBER", right_on="location").drop(axis=1,
#                                                                                                         columns=[
#                                                                                                             'location'])
#     sort_index = data.set_index(["FLIGHT_ID"]).sort_values(by=[weather], ascending=False, axis=0).index.values
#     sort_index.tolist()
#     duplicate_index = []
#     for index in sort_index.tolist():
#         if index not in duplicate_index:
#             duplicate_index.append(index)
#     data_to_excel = data.set_index(["FLIGHT_ID", "FIRST_STAND_NUMBER"]).ix[duplicate_index]
#     FIRST_STAND_NUMBER_FLIGHT_ID[weather] = data_to_excel
#
# ### 保存数据
# with pd.ExcelWriter('./data/重复航班停机坪统计.xlsx') as writer:
#     for weather,df in FIRST_STAND_NUMBER_FLIGHT_ID.items():
#         df.to_excel(writer, sheet_name=weather)


### 以起飞或降落，航班跑道为维度，统计原机位安排的分布状况

FIRST_STAND_NUMBER_ARR_DEP_RUN_WAY = {}
for state in arr_dep:
    for way in run_ways:
        for weather in weathers:
            sql = """
          select FIRST_STAND_NUMBER,count(*)  as {0}_{1}_{2} from {2}
          where (FIRST_STAND_NUMBER is not NULL AND  FIRST_STAND_NUMBER <> " ")
          and  ARR_DEP = \"{0}\"
          and R_RUNWAY = \"{1}\"
          group by FIRST_STAND_NUMBER
          ORDER BY COUNT(*) DESC;
            """.format(state, way, weather)
            name = "{}_{}_{}".format(state, way, weather)
            FIRST_STAND_NUMBER_ARR_DEP_RUN_WAY[name] = pd.read_sql(sql, engine)
### 保存数据
with pd.ExcelWriter('./data/飞机起飞降落-跑道位置与停机坪位置安排.xlsx') as writer:
    for key, values in FIRST_STAND_NUMBER_ARR_DEP_RUN_WAY.items():
        values['RATIO'] = values[key] / values[key].sum() * 100
        data = values.merge(apron_region, how='left', left_on="FIRST_STAND_NUMBER", right_on="location")
        data.drop(axis=1, columns=['location'], inplace=True)
        data.to_excel(writer, sheet_name=key)
pass
