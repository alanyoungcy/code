#  Created at 2019/3/22                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com

from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import  TensorBoard
from keras import layers
from keras import models
from keras.layers import LSTM
import os

def cnn_stand_():
    # 调整学习速率
    def scheduler(epoch):
        if epoch != 0 and epoch % 2 == 0:
            # 每隔2个epoch就调整一次学习率,
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to %d", lr * 0.1)
        return K.get_value(model.optimizer.lr)

    # 构建学习率回调函数句柄
    reduce_lr = LearningRateScheduler(scheduler)

    # 使用tensorbaord 观察训练结果
    borad = TensorBoard(log_dir='./logs', histogram_freq=0,
                        batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                        embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    # 如果验证损失下降， 那么在每个训练轮之后保存模型。
    checkpointer = ModelCheckpoint(filepath='./model_output/loss_min.hdf5', verbose=1, save_best_only=True)

    # 加载处理后的数据进行模型训练
    train_test_dataset = np.load("./dataset/train_test_dataset_stand.npz")

    # 原数据维度963，为了能够使用卷积神经网络，我们将数据维度弄成960
    # 在折叠二维（30，32）


    X_train = train_test_dataset['X_train']
    zeros=np.zeros((X_train.shape[0],10))
    X_train=np.c_[X_train,zeros]
    Y_train_last_stand = train_test_dataset['Y_train_last_stand']
    X_test = train_test_dataset['X_test']
    zeros = np.zeros((X_test.shape[0], 10))
    X_test = np.c_[X_test, zeros]
    Y_test_last_stand = train_test_dataset['Y_test_last_stand']

    print(X_train.shape)
    input_x=16
    input_y=20

    # 对数据进行堆叠操作
    X_train = np.column_stack((X_train, X_train, X_train))
    X_test = np.column_stack((X_test, X_test, X_test))

    # 为使用卷积神经网络，改变数据维度，一维变成二维
    X_train = np.reshape(X_train, (X_train.shape[0], input_x, input_y, 3))
    X_test = np.reshape(X_test, (X_test.shape[0], input_x, input_y, 3))

    # 对标签进行对热编码，存在测试集和训练集标签不匹配的情况，强制设定为163
    Y_train_last_stand = to_categorical(Y_train_last_stand, 163)
    Y_test_last_stand = to_categorical(Y_test_last_stand, 163)


    # 构建深度神经网络
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu',
                            input_shape=(input_x, input_y, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (2, 2), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (2, 2), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    # 标签属性类别有163个，最后一层应用softmax
    model.add(layers.Dense(163, activation='softmax'))
    # 输出网络层次
    model.summary()

    #
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # 模型训练,batch_size,nb_epoch根据自己的硬件需求调整。
    model.fit(X_train, Y_train_last_stand, batch_size=1024, nb_epoch=10, validation_data=(X_test, Y_test_last_stand),
              callbacks=[reduce_lr, borad, checkpointer])
    # 保存模型

    model.save('./model_output/cnn_stand.h5')
    print("训练完成，模型已保存在 model_output/cnn_stand.h5 ")

if __name__=="__main__":
    # main()
    pass
