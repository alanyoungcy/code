
## 运行工程
```shell
python3 main.py
```
## ipython 
用jupyter打开的文件，主要用于展现数据清洗，数据建模的过程

data_rangle.ipynb   为现阶段以天气状态为维度对机位和闸口做的相关数据分析。

ml_flight_id_diffrent_data_source.ipynb  机位改变分析与建模描述notebook。

runway_apron.ipynb 对航班与跑道进行分析

## data_wrangling
用于数据清理的脚本

aircraft_position.py  飞机机位与区域关联脚本

heatmap.py 在地图上标示机位标号已经数值热力值脚本

read_sql.py    与mysql进行数据读取与分析的脚本

regional_division.py 对机位区域进行编码

## dataset
用于存放中间过程的数据

column_to_class.json   存储特征数据总类别属性，用于不同天气状态下训练特征数据，保持总的特征维度不变。

## model_out
用于存放建模结果的目录，包括预测指标，模型文件

model_optimization_output.txt   各个分类器训练结果
out.txt 分类器训练结果 
## model_train_gate_change
用于预测航班最总位置改变可能性

model_fit.py   新数据使用训练好的模型进行预测 

model_train.py 模型训练脚本

model_optimization.py 运用多进程同时使用多个分类器对数据进行训练，比较训练结果

dnn_model.py  使用全连接神经网络，对数据进行训练预测，准确率为97%

cnn.py        使用卷积神经网络，对数据进行训练预测，准确率达到99%,比任何模型准确率都高。

## model_train_last_stand
用于预测航班最总位置,航班最终位置类别总共有160个。

cnn_stand.py    使用卷积神经网络，对数据进行训练预测，准确率达到84%，仅需要2个epoch就可以达到

dnn_stand.py     使用全连接神经网络，对数据进行训练预测，准确率为84%，需要10个epoch

model_train_stand.py  预测航班最终位置所需要的数据预处理




