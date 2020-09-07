#  Created at 2019/4/10                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com



import numpy as np
import pandas as pd
from keras.models import load_model
import sys
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from model_train_gate_chage.model_train_gate_change import tranform, model_assessment, data_wrangle
import json
from sklearn.externals import joblib
import time

import json
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report

def model_predict_stand_(weathers="all", sql=None):
    """
        对测试数据进行训练
        @return:
        """
    if weathers == "all":
        weathers = ["AODB_MAJOR", "Strong_Wind", "Thunder_Storm", "abnormal_time", "normal_time"]  # 航班所处天气状态
    # weathers = [ "abnormal_time"]  # 航班所处天气状态
    for weather in weathers:
        print(weather)
        data_DEF_1 = data_wrangle(weather, fit=True, sql=sql)

        ## 在本次建模中可以采用的类别类型属性
        category_columns = ["FLIGHT_ID", "R_RUNWAY", "STD_STAND_APRON", "FIRST_STAND_NUMBER",
                            "LAST_STAND_NUMBER", "FLIGHT_TYPE", "A_FLIGHT_AIRLINE_ID",
                            "HA_CARGO_AGENT", "AP_ORIGIN_DEST", "HA_PAX_AGENT", "HA_MAINTENANCE_AGENT",
                            "FSTAT_AIRSIDE_STATUS", "FSTAT_TIME_STATUS", "HA_GROUND_AGENT",
                            "FIRST_STAND_NUMBER_APRON",
                            'ST_Scheduled', "ST_Passenger", "ST_Cargo", "ST_Mail", "ST_Charter", 'WTC']

        series_columns = ['maxPassengers', 'Speed(mph)', 'Service_Ceiling(ft)', 'Range(NMi)',
                          'OEW(lbs)', 'MTOW(lbs)', 'Wing_Span(ft)',
                          'Wing_Area(ft2)', 'Length(ft)', 'Height(ft)']
        ### 去除空值
        data_DEF_2 = data_DEF_1[category_columns + series_columns].dropna(how="any", axis=0)

        with open("./dataset/column_to_class_stand.json", 'r') as f:
            # 将之前在训练时，独热编码后的信息加载进来
            column_to_class_stand = json.loads(f.read())

        # 把与LAST_STAND_NUMBER存在相关性的属性保留
        relation_columns = column_to_class_stand['relation_columns']

        # 分割标签属性和特征属性
        label_columns = ["LAST_STAND_NUMBER"]

        # # 分割数据属性和标签属性
        for label in label_columns:
            relation_columns.remove(label)
        data_columns = relation_columns


        standarlize = StandardScaler()
        for column in series_columns:
            data_DEF_2[column] = standarlize.fit_transform(data_DEF_2[[column]]).tolist()

        # 先对特征属性进行独热编码
        # ordinal_encoder = OrdinalEncoder()
        cat_encoder = LabelBinarizer()
        for column in data_columns:
            # 维持与训练数据的独热编码后的数据属性维度相同
            encoder = cat_encoder.fit(column_to_class_stand[column])
            data_DEF_2[column] = encoder.transform(data_DEF_2[column]).tolist()

        data_columns.extend(series_columns)

        # 分割标签数据集和特征数据集的训练集和测试集
        X_test = data_DEF_2[data_columns]
        Y_test_gate_change = data_DEF_2[["LAST_STAND_NUMBER"]].copy()

        # 对特征数据集进行进一步的数据转化
        X_test = tranform(X_test)
        print("测试集数据维度{}".format(X_test.shape))

        # 对标签属性数据进行独热编码
        ordinal_encoder = OrdinalEncoder()
        Y_test_last_stand = ordinal_encoder.fit_transform(Y_test_gate_change)

        test_set_no_one_hot = pd.read_csv("./dataset/test_set_no_one_hot.csv")
        # train_test_dataset = np.load("./dataset/train_test_dataset_stand.npz")
        # X_test = train_test_dataset['X_test']
        #


        zeros=np.zeros((X_test.shape[0],10))
        X_test=np.c_[X_test,zeros]
        # Y_test_last_stand = train_test_dataset['Y_test_last_stand']
        # 对数据进行堆叠操作
        X_test = np.column_stack((X_test, X_test, X_test))


        with open("./dataset/column_to_class_stand.json", 'r') as f:
            # 将之前在训练时，独热编码后的信息加载进来
            column_to_class_stand = json.loads(f.read())

        Y_test_last_stand_categories=column_to_class_stand['LAST_STAND_NUMBER']
        ordinal_encoder = OrdinalEncoder()
        ordinal_encoder.fit_transform(pd.DataFrame(Y_test_last_stand_categories))

        input_x=16
        input_y=20

        # 为使用卷积神经网络，改变数据维度，一维变成二维
        X_test = np.reshape(X_test, (X_test.shape[0], input_x, input_y, 3))

        model=load_model("./model_output/cnn_stand.h5")
        X_test_predict=np.argmax(model.predict(X_test), 1)

        X_test_predict=X_test_predict.reshape((X_test_predict.shape[0],1))

        LAST_STAND_PREDICT=ordinal_encoder.inverse_transform(X_test_predict)
        # LAST_STAND_TEST=ordinal_encoder.inverse_transform(Y_test_last_stand)
        # test_set_no_one_hot['LAST_STAND_TEST']=LAST_STAND_TEST
        # test_set_no_one_hot['LAST_STAND_PREDICT']=LAST_STAND_PREDICT
        # test_set_no_one_hot["STATUS"] = test_set_no_one_hot["LAST_STAND_TEST"] == test_set_no_one_hot["LAST_STAND_PREDICT"]

        # labels_list=np.array(list(set(Y_test_last_stand.flatten())| set(X_test_predict.flatten())))
        # labels_list=labels_list.reshape((labels_list.shape[0],1))
        # labels=ordinal_encoder.inverse_transform(labels_list).flatten().tolist()
        #
        #
        # output_dict = classification_report(Y_test_last_stand, X_test_predict,
        #                                     target_names=labels, output_dict=True)
        # output_df=pd.DataFrame.from_dict(output_dict).T
        # print(output_dict)

        # with pd.ExcelWriter('./dataset/test_set_no_one_hot.xlsx') as writer:  # doctest: +SKIP
        #     test_set_no_one_hot.to_excel(writer, sheet_name='数据')
        #     output_df.to_excel(writer,sheet_name="预测结果")
        #     print("数据保存成功，dataset/test_set_no_one_hot.xlsx")

    return LAST_STAND_PREDICT
