#  Created at 2019/4/10                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com
from model_train_gate_chage.model_train_gate_change import model_train_gate_change_
from model_train_gate_chage.model_predict_gate_change import model_predict_gate_change_
from model_train_last_stand.model_train_stand import  model_train_stand_
from model_train_last_stand.model_predict_stand import model_predict_stand_


class model_pipeline():
    # def __init__(self,obj):
    #     assert obj in ["gate_change", "last_stand"], \
    #         "请输入您要预测的目标，gate_change or last_stand"
    #     self.obj=obj

    def __call__(self, obj):
        assert obj in ["gate_change", "last_stand"], \
            "请输入您要预测的目标，gate_change or last_stand"
        self.obj = obj

    def model_train(self,sql=None,algorithm=None):
        train={}
        train["gate_change"]=model_train_gate_change_
        train["last_stand"]=model_train_stand_
        train[self.obj]()
    def model_predict(self,weathers="all"):
        predict={}
        predict["gate_change"]= model_predict_gate_change_
        predict["last_stand"]=model_predict_stand_
        return predict[self.obj](weathers)


