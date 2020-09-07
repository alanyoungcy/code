#  Created at 2019/3/29                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com
import numpy as np
import pandas as pd
from keras.models import load_model


import json
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report

def model_fit_stand_():
    test_set_no_one_hot = pd.read_csv("./dataset/test_set_no_one_hot.csv")
    train_test_dataset = np.load("./dataset/train_test_dataset_stand.npz")
    X_test = train_test_dataset['X_test']
    zeros=np.zeros((X_test.shape[0],10))
    X_test=np.c_[X_test,zeros]
    Y_test_last_stand = train_test_dataset['Y_test_last_stand']
    # 对数据进行堆叠操作
    X_test = np.column_stack((X_test, X_test, X_test))


    with open("./dataset/stand_column_to_class.json", 'r') as f:
        # 将之前在训练时，独热编码后的信息加载进来
        column_to_class = json.loads(f.read())

    Y_test_last_stand_categories=column_to_class['LAST_STAND_NUMBER']
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
    LAST_STAND_TEST=ordinal_encoder.inverse_transform(Y_test_last_stand)
    test_set_no_one_hot['LAST_STAND_TEST']=LAST_STAND_TEST
    test_set_no_one_hot['LAST_STAND_PREDICT']=LAST_STAND_PREDICT
    test_set_no_one_hot["STATUS"] = test_set_no_one_hot["LAST_STAND_TEST"] == test_set_no_one_hot["LAST_STAND_PREDICT"]

    labels_list=np.array(list(set(Y_test_last_stand.flatten())| set(X_test_predict.flatten())))
    labels_list=labels_list.reshape((labels_list.shape[0],1))
    labels=ordinal_encoder.inverse_transform(labels_list).flatten().tolist()


    output_dict = classification_report(Y_test_last_stand, X_test_predict,
                                        target_names=labels, output_dict=True)
    output_df=pd.DataFrame.from_dict(output_dict).T
    # print(output_dict)

    with pd.ExcelWriter('./dataset/test_set_no_one_hot.xlsx') as writer:  # doctest: +SKIP
        test_set_no_one_hot.to_excel(writer, sheet_name='数据')
        output_df.to_excel(writer,sheet_name="预测结果")
        print("数据保存成功，dataset/test_set_no_one_hot.xlsx")

    return X_test_predict











