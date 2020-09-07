#  Created at 2019/3/22                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com        

import pandas as pd
import numpy as np
from data_wrangling.aircraft_position import apron_region
from scipy import stats
import itertools
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import copy
import json
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from model_train_gate_chage.data_transform import category_reduction,flight_type_transfrom,server_transform

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

engine = create_engine('mysql+pymysql://hkia_v1:Asdf168!!@192.168.10.170:3306/HKIA_v1')


def tranform(data_tranform):
    """
    对特征属性列进行转换
    :param data_tranform:
    :return:
    """
    tmp = []
    print(data_tranform.shape)
    for i in range(data_tranform.shape[0]):
        data = []
        for j in range(data_tranform.shape[1]):
            data.extend(data_tranform.iloc[i, j])
        tmp.append(data)
    return np.array(tmp)


def plot_roc_curve(Y_test, y_predict, label=None):
    """
    绘制ROC曲线
    :param Y_test: 测试集特征属性值
    :param y_predict: 测试集预测结果
    :param label:
    :return: None
    """
    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(Y_test, y_predict)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.show()


def model_assessment(model, X_test, Y_test):
    """
    对训练后的模型进行评估
    :param model:训练后的模型
    :param X_test: 测试数据特征值
    :param Y_test: 测试数据标签值
    :return: 评估结果输出到屏幕
    """
    ### 模型评估
    print("测试集正确率：{}%".format(model.score(X_test, Y_test) * 100))
    y_predict = model.predict(X_test)

    #### 绘制ROC曲线
    # plot_roc_curve(Y_test, y_predict, label=None)

    ### 输出报告
    print(classification_report(Y_test, y_predict))


def data_wrangle(weather,fit=None):
    """
    :param weather: 按天气类别来获取航班数据
    :return: 返回数据处理后的数据
    """

    #### 将所有数据都选取
    ### 属性选取，根据缺失程度，与飞机停机位关联关系。
    sql = """
    select FLIGHT_ID,R_RUNWAY,GATE_OPEN_DATE,GATE_CLOSE_DATE,SCHEDULED_DATE,
    ESTIMATED_DATE,ACTUAL_DATE,STD_STAND,NO_OF_STAND_CHANGES,FIRST_STAND_NUMBER,LAST_STAND_NUMBER,FLIGHT_TYPE,
    IC_GATE_CODE,A_FLIGHT_AIRLINE_ID,ARR_DEP,AT_AIRCRAFT_TYPE,HA_CARGO_AGENT,AP_ORIGIN_DEST,IC_CHECKIN_SUMMARY_CODE,
    FSTAT_GATE_STATUS,FSTAT_AIRSIDE_STATUS,ST_SERVICE_TYPE,INTERNATIONAL_DOMESTIC,HA_PAX_AGENT,
    HA_MAINTENANCE_AGENT,AIRCRAFT_REG_GUESS,AT_SUBTYPE,FSTAT_TIME_STATUS,HA_GROUND_AGENT,AC_AIRCRAFT
    from {}
    """.format(weather)
    data = pd.read_sql_query(sql, engine)

    #### 由于从SQL里面读取的数据，对于部分空缺值会采用‘’代替，因此可能对后续分析造成影响
    data.replace('', pd.NaT, inplace=True)

    # data
    ### 只选取起飞航班数据
    # data_DEF=data[data.ARR_DEP=="D"]
    data_DEF = data
    ### 去除属性缺失值较多的列
    data_DEF_1 = data_DEF.drop(
        labels=["GATE_OPEN_DATE", "GATE_CLOSE_DATE", "ESTIMATED_DATE", "IC_GATE_CODE", "IC_CHECKIN_SUMMARY_CODE",
                "FSTAT_GATE_STATUS"], axis=1)

    #### 查看上表，我们发现数据缺失情况不是很严重，我们对于哪些数据有缺失的，直接删除即可
    data_DEF_1.dropna(axis=0, how="any", inplace=True)

    # #### 构造标签属性，False 表示预定机位位置安排未改变，True 表示预定机位位置安排改变
    # data_DEF_1['gate_change_status'] = data_DEF_1.FIRST_STAND_NUMBER != data_DEF_1.LAST_STAND_NUMBER

    #### FLIGHT_ID有两千多个类别，只保留名称前三位作为训练
    # data_DEF_1.FLIGHT_ID.value_counts()
    # data_DEF_1['FLIGHT_ID_H3'] = data_DEF_1.FLIGHT_ID.str[:3]

    #### 对R_RUNWAY不做处理
    # data_DEF_1.R_RUNWAY.value_counts()

    ### 以FIRST_STAND_NUMBER,进行区域划分
    data_DEF_1['FIRST_STAND_NUMBER_APRON'] = \
    data_DEF_1.merge(apron_region, how='left',
                     left_on="FIRST_STAND_NUMBER", right_on="location")["APRON"].values

    if fit == None:
        # data_category_rudection存储 属性缩减后的key_value
        data_category_rudection = {}

        # FLIGHT_ID有两千多个类别，缩减类别
        # data_DEF_1['FLIGHT_ID_H3'] = data_DEF_1.FLIGHT_ID.str[:3]

        data_category_rudection["FLIGHT_ID"] = category_reduction(data_DEF_1, "FLIGHT_ID")

        # AP_ORIGIN_DEST 类别比较多，进行类别缩减
        data_category_rudection["AP_ORIGIN_DEST"] = category_reduction(data_DEF_1, "AP_ORIGIN_DEST")

        # A_FLIGHT_AIRLINE_ID 类别比较多，进行类别缩减
        data_category_rudection["A_FLIGHT_AIRLINE_ID"] = category_reduction(data_DEF_1, "A_FLIGHT_AIRLINE_ID")

        # FIRST_STAND_NUMBER ，以类别变量处理,进行类别变量缩减。
        # data_DEF_1.FIRST_STAND_NUMBER.value_counts()

        # data_category_rudection["FIRST_STAND_NUMBER"] = category_reduction(data_DEF_1, "FIRST_STAND_NUMBER")

        with open("./dataset/data_category_rudection_stand.json", 'w', encoding='utf-8') as file:
            file.write(json.dumps(data_category_rudection))

    if fit == True:
        with open("./dataset/data_category_rudection_stand.json", 'r', encoding='utf-8') as file:
            data_category_rudection=json.loads(file.read())

        df_category_rudection=pd.DataFrame.from_dict(data_category_rudection['FLIGHT_ID'])
        data_DEF_1['FLIGHT_ID']. \
            apply(lambda data:df_category_rudection[df_category_rudection["FLIGHT_ID"]==data] \
                     .group.values[0])

    ### 以STD_STAND,进行区域划分
    data_DEF_1['STD_STAND_APRON'] = \
    data_DEF_1.merge(apron_region, how='left', left_on="STD_STAND", right_on="location")["APRON"].values

    ### NO_OF_STAND_CHANGES零和非零比例差别，将非零比例全化为1。
    data_DEF_1.NO_OF_STAND_CHANGES.value_counts(normalize=True)
    data_DEF_1.NO_OF_STAND_CHANGES.replace(to_replace=[2, 3, 4, 5, 11], value=1, inplace=True)
    data_DEF_1.NO_OF_STAND_CHANGES.value_counts(normalize=True)

    #### FIRST_STAND_NUMBER ，以类别变量处理。
    # data_DEF_1.FIRST_STAND_NUMBER.value_counts()

    ####  以LAST_STAND_NUMBER,进行区域划分
    data_DEF_1['LAST_STAND_NUMBER_APRON'] = \
    data_DEF_1.merge(apron_region, how='left', left_on="LAST_STAND_NUMBER", right_on="location")["APRON"].values

    index = data_DEF_1.query("FSTAT_AIRSIDE_STATUS in ['L','M']").index
    data_DEF_1.drop(axis=0, index=index, inplace=True)

    # 串接属性
    ## 串接飞机类型
    data_DEF_1=flight_type_transfrom(data_DEF_1)
    ## 串接服务
    data_DEF_1 = server_transform(data_DEF_1)

    return data_DEF_1


def model_train_stand_():
    ### 数据获取与清洗
    data_DEF_1 = data_wrangle("AODB_MAJOR")

    ## 在本次建模中可以采用的类别类型属性
    category_columns = ["FLIGHT_ID", "R_RUNWAY", "STD_STAND_APRON", "FIRST_STAND_NUMBER",
                        "LAST_STAND_NUMBER", "FLIGHT_TYPE", "A_FLIGHT_AIRLINE_ID",
                        "HA_CARGO_AGENT", "AP_ORIGIN_DEST", "HA_PAX_AGENT", "HA_MAINTENANCE_AGENT",
                        "FSTAT_AIRSIDE_STATUS", "FSTAT_TIME_STATUS", "HA_GROUND_AGENT",
                        "FIRST_STAND_NUMBER_APRON",
                        'ST_Scheduled', "ST_Passenger", "ST_Cargo", "ST_Mail", "ST_Charter", 'WTC']

    series_columns =['maxPassengers', 'Speed(mph)','Service_Ceiling(ft)', 'Range(NMi)',
             'OEW(lbs)', 'MTOW(lbs)', 'Wing_Span(ft)',
                 'Wing_Area(ft2)', 'Length(ft)', 'Height(ft)']


    ### 去除空值
    data_DEF_2 = data_DEF_1[category_columns+series_columns].dropna(how="any", axis=0)


    ### 卡方检验
    chi_data = {"column_1": [], "column_2": [], "P_value": [], "relation": []}
    for i, j in itertools.product(list(category_columns), repeat=2):
        cross_table = pd.crosstab(data_DEF_1[i], data_DEF_1[j])
        P_value = stats.chi2_contingency(cross_table)[1]
        chi_data["column_1"].append(i)
        chi_data["column_2"].append(j)
        chi_data["P_value"].append(P_value)
        chi_data["relation"].append(P_value < 0.001)

    chi_df = pd.DataFrame(chi_data)
    df_corelation = chi_df.pivot_table(values="relation", index="column_1", columns="column_2")

    #### 把与gate_change_status存在相关性的属性保留
    relation_columns = list(df_corelation[df_corelation.LAST_STAND_NUMBER == True].LAST_STAND_NUMBER.index)
    print("相关属性列：{}".format(category_columns))

    column_to_class_stand = {}
    column_to_class_stand["relation_columns"] = copy.deepcopy(relation_columns)

    #### 分割标签属性和特征属性
    label_columns = ["LAST_STAND_NUMBER"]
    for label in label_columns:
        relation_columns.remove(label)
    data_columns = relation_columns

    from sklearn.preprocessing import StandardScaler
    standarlize=StandardScaler()
    for column in series_columns:
        data_DEF_2[column]=standarlize.fit_transform(data_DEF_2[[column]]).tolist()


    # 先对特征属性进行独热编码
    cat_encoder = LabelBinarizer()

    for column in data_columns:
        print(column)
        column_to_class_stand[column] = cat_encoder.fit(data_DEF_2[column]).classes_.tolist()  ### 保存独热编码后的转化特征属性模型
        data_DEF_2[column] = cat_encoder.fit_transform(data_DEF_2[column]).tolist()
        pass

    # 划分训练集和测试集，按标签进行分层次采样
    train_set, test_set = train_test_split(data_DEF_2, test_size=0.1, random_state=66)

    data_columns.extend(series_columns)

    # 航班停机位测试数据（未独热编码）
    test_set_no_one_hot=data_DEF_1.loc[test_set.index]
    test_set_no_one_hot.to_csv("./dataset/test_set_no_one_hot.csv", )

    # 分割标签数据集和特征数据集的训练集和测试集
    X_test = test_set[data_columns]
    X_train = train_set[data_columns]  # drop labels for training set
    Y_test_last_stand = test_set[["LAST_STAND_NUMBER"]].copy()
    Y_train_last_stand = train_set[["LAST_STAND_NUMBER"]].copy()
    duplicates_label=data_DEF_2[["LAST_STAND_NUMBER"]].drop_duplicates("LAST_STAND_NUMBER").copy()


    # 对特征数据集进行进一步的数据转化
    X_test = tranform(X_test)
    X_train = tranform(X_train)
    print("训练集维度: {}".format(X_train.shape))
    print("测试集维度：{}".format(X_test.shape))

    # 对标签数据进行独热编码
    ordinal_encoder = OrdinalEncoder()
    column_to_class_stand['LAST_STAND_NUMBER']=ordinal_encoder.fit(duplicates_label).categories_[0].tolist()
    Y_train_last_stand = ordinal_encoder.transform(Y_train_last_stand)
    Y_test_last_stand= ordinal_encoder.transform(Y_test_last_stand)


    with open("./dataset/column_to_class_stand.json", 'w', encoding='utf-8') as file:
        file.write(json.dumps(column_to_class_stand))


    # 对处理后的数据集进行保存，便于模型优化时数据再读取
    np.savez("./dataset/train_test_dataset_stand.npz", X_train=X_train,
             Y_train_last_stand=Y_train_last_stand,
             X_test=X_test, Y_test_last_stand=Y_test_last_stand)
    print("数据集保存成功")


if __name__ == "__main__":
    pass
