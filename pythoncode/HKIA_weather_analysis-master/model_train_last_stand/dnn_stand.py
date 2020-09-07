#  Created at 2019/3/22                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.callbacks import LearningRateScheduler
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import os


def dnn_stand_():
    # 调整学习速率
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
    # 用tensorboard记录训练过程
    borad = TensorBoard(log_dir='./logs', histogram_freq=0,
                        batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                        embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                        update_freq='epoch')
    # 获取数据
    train_test_dataset = np.load("./dataset/train_test_dataset_stand.npz")
    X_train = train_test_dataset['X_train']
    Y_train_last_stand = train_test_dataset['Y_train_last_stand']
    X_test = train_test_dataset['X_test']
    Y_test_last_stand = train_test_dataset['Y_test_last_stand']

    # 标签数据独热编码
    labels = to_categorical(Y_train_last_stand, 163)
    labels_test = to_categorical(Y_test_last_stand, 163)


    input_shape=(X_train.shape)[1]
    # 构建网络
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, kernel_regularizer=regularizers.l2(0.001), init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(163, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(X_train, labels, batch_size=512, nb_epoch=60, validation_data=(X_test, labels_test),
                        callbacks=[reduce_lr, borad])

    # 模型保存
    model.save('./model_output/dnn_stand.h5')
    print("模型已保存至model_output/dnn_stand.h5")
    print(163)


if __name__ == "__main__":
    pass
