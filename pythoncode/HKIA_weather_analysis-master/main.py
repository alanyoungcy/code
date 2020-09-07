#  Created at 2019/3/28                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com
from model_train_gate_chage.model_train_gate_change import model_train_gate_change_
from model_train_gate_chage.model_predict_gate_change import model_predict_gate_change_
from model_train_gate_chage.model_optimization import pool_train
from model_train_gate_chage.dnn_gate_change import dnn_gate_change_
from model_train_gate_chage.cnn_gate_change import cnn_gate_change_
from model_train_last_stand.model_train_stand import model_train_stand_
from model_train_last_stand.dnn_stand import dnn_stand_
from model_train_last_stand.cnn_stand import cnn_stand_
from model_train_last_stand.model_fit_stand import model_fit_stand_
from model_train_gate_chage.data_transform import *
# from model_train_gate_chage.model_train_t import model_train_gate_change_
from model_api.web_server import app

from model_api.model_pipeline import model_pipeline


import warnings
warnings.filterwarnings('ignore')
import os
print(os.getcwd())
os.system("export PYTHONPATH={}".format(os.getcwd()))

def main():

    print("*"*40*2)
    print("航班机位改变可能性模型训练")
    model_train_gate_change_()
    print("训练完成")

    print("*" * 40*2)
    print("航班机位改变测试数据预测评估")
    model_predict_gate_change_()
    print("测试数据预测评估完成")
    pass

    print("*" * 40*2)
    print("多线程训练多分类器")
    pool_train()
    print("训练完成完成")


    print("*" * 40*2)
    print("使用全连接深度神经网络对航班机位改变数据属性进行训练")
    dnn_gate_change_()
    print("训练完成")

    print("*" * 40 * 2)
    print("使用卷积神经网络对航班机位改变数据属性进行训练")
    cnn_gate_change_()
    print("训练完成")

    print("*" * 40*2)
    print("航班最终机位数据属性特征提取与转化")
    model_train_stand_()
    print("数据转化完成")

    print("*" * 40 * 2)
    print("使用全连接深度神经网络对航班最终机位数据属性进行训练")
    dnn_stand_()
    print("训练完成")

    print("*" * 40 * 2)
    print("使用卷积神经网络对航班最终机位数据属性进行训练")
    cnn_stand_()
    print("训练完成")


if __name__=="__main__":
    cnn_stand_()
    
    # model_fit_stand_()
    # pipeline=model_pipeline()
    # pipeline("last_stand")
    # pipeline.model_train()
    # print(pipeline.model_predict(weathers=["abnormal_time"]))
    # app.run()
    pass

