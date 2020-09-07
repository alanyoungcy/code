from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
import numpy as np
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import os


def dnn_gate_change_():
    ### 调整学习速率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001,
                                  cooldown=0, min_lr=0)

    borad = TensorBoard(log_dir='./logs', histogram_freq=0,
                        batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                        embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                        update_freq='epoch')
    input_shape=146
    # 加载数据集
    train_test_dataset = np.load("./dataset/train_test_dataset.npz")

    # 提取训练集
    X_train = train_test_dataset['X_train']
    Y_train_gate_change = train_test_dataset['Y_train_gate_change']

    # 提取测试集
    X_test = train_test_dataset['X_test']
    Y_test_gate_change = train_test_dataset['Y_test_gate_change']




    # 构建全连接神经网络
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, kernel_regularizer=regularizers.l2(0.001),
                    init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.summary()
    model.fit(X_train, Y_train_gate_change, batch_size=256, nb_epoch=10,
                        validation_data=(X_test, Y_test_gate_change), callbacks=[reduce_lr, borad])

    # 模型保存
    model.save('./model_output/dnn_gate_change.h5')
    print("模型已训练完成，保存在model_output/dnn_gate_change.h5")

if __name__ == "__main__":
    pass
    # main()
