#  Aothor: Ethan Lee
#  E-mail: ethan2lee.coder@gmail.com
import warnings
warnings.filterwarnings('ignore')
import sys
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from model_train_gate_chage.model_train_gate_change import tranform, model_assessment, data_wrangle
import json
from sklearn.externals import joblib
import time


def test_set_assessment(modol, weather, data, X_test, Y_test, output_file):
    """
    保存训练集预测结果

    :param modol: 经过训练集训练好的模型
    :param weather: 根据天气状态，选择航班数据
    :param data: 输入的数据集，用于查看数据集数量
    :param X_test: 测试集特征属性数据集
    :param Y_test: 测试集标签属性数据集
    :param output_file: 保存结果的文件名称
    :return:
    """
    savedStdout = sys.stdout  # 保存标准输出流
    with open(output_file, 'a+', encoding="utf-8") as file:
        sys.stdout = file  # 标准输出重定向至文件
        print('*' * 40 * 3)
        print("the predict result of dataset {}".format(weather))
        print(time.ctime())
        print("总数据量：{}".format(len(data)))
        ### 进行模型评估
        model_assessment(modol, X_test, Y_test)
    sys.stdout = savedStdout  # 恢复标准输出流

def model_predict_gate_change_(weathers="all", sql=None):
    """
    对测试数据进行训练
    @return:
    """
    if weathers == "all":
        weathers = ["AODB_MAJOR","Strong_Wind", "Thunder_Storm", "abnormal_time", "normal_time"] # 航班所处天气状态
    # weathers = [ "abnormal_time"]  # 航班所处天气状态
    for weather in weathers:
        print(weather)
        data_DEF_1 = data_wrangle(weather,fit=True,sql=sql)

        ## 在本次建模中可以采用的类别类型属性
        category_columns = ["FLIGHT_ID", "R_RUNWAY", "STD_STAND_APRON", "FIRST_STAND_NUMBER",
                            "gate_change_status", "FLIGHT_TYPE", "A_FLIGHT_AIRLINE_ID",
                            "HA_CARGO_AGENT", "AP_ORIGIN_DEST", "HA_PAX_AGENT", "HA_MAINTENANCE_AGENT",
                            "FSTAT_AIRSIDE_STATUS", "FSTAT_TIME_STATUS", "HA_GROUND_AGENT",
                            "FIRST_STAND_NUMBER_APRON",
                            'ST_Scheduled', "ST_Passenger", "ST_Cargo", "ST_Mail", "ST_Charter", 'WTC']

        series_columns = ['maxPassengers', 'Speed(mph)', 'Service_Ceiling(ft)', 'Range(NMi)',
                          'OEW(lbs)', 'MTOW(lbs)', 'Wing_Span(ft)',
                          'Wing_Area(ft2)', 'Length(ft)', 'Height(ft)']
        ### 去除空值
        data_DEF_2 = data_DEF_1[category_columns+series_columns].dropna(how="any", axis=0)

        with open("./dataset/column_to_class.json", 'r') as f:
            # 将之前在训练时，独热编码后的信息加载进来
            column_to_class = json.loads(f.read())

        # 把与gate_change_status存在相关性的属性保留
        relation_columns = column_to_class['relation_columns']

        # 分割标签属性和特征属性
        label_columns = ["gate_change_status"]

        # 分割数据属性和标签属性
        for label in label_columns:
            relation_columns.remove(label)
        data_columns = relation_columns

        from sklearn.preprocessing import StandardScaler
        standarlize = StandardScaler()
        for column in series_columns:
            data_DEF_2[column] = standarlize.fit_transform(data_DEF_2[[column]]).tolist()

        # 先对特征属性进行独热编码
        # ordinal_encoder = OrdinalEncoder()
        cat_encoder = LabelBinarizer()
        for column in data_columns:
            #维持与训练数据的独热编码后的数据属性维度相同
            encoder = cat_encoder.fit(column_to_class[column])
            data_DEF_2[column] = encoder.transform(data_DEF_2[column]).tolist()

        data_columns.extend(series_columns)

        # 分割标签数据集和特征数据集的训练集和测试集
        X_test = data_DEF_2[data_columns]
        Y_test_gate_change = data_DEF_2[["gate_change_status"]].copy()

        # 对特征数据集进行进一步的数据转化
        X_test = tranform(X_test)
        print("测试集数据维度{}".format(X_test.shape))

        # 对标签属性数据进行独热编码
        ordinal_encoder = OrdinalEncoder()
        Y_test_gate_change = ordinal_encoder.fit_transform(Y_test_gate_change)

        #### 模型导入
        sgd_clf_gate_chage = joblib.load("./model_output/sgd_clf_gate_chage.m")

        ### 测试数据评估
        test_set_assessment(sgd_clf_gate_chage, weather, data_DEF_2, X_test, Y_test_gate_change, "./model_output/out.txt")
        print("预测结果保存在model_output/out.txt中，请前往查看")
        return sgd_clf_gate_chage.predict_proba(X_test)


if __name__ == "__main__":
    model_predict_gate_change_()
