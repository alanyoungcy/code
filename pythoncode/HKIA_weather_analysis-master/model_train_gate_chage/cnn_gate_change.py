#  Created at 2019/3/21                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com        


import numpy as np
from keras.callbacks import ReduceLROnPlateau, TensorBoard
import os
from keras.callbacks import LearningRateScheduler
import keras.backend as K



def cnn_gate_change_():

    # 调整学习速率
    def scheduler(epoch):
        if epoch != 0 and epoch % 2 == 0:
            # 每隔2个epoch就调整一次学习率,
            lr = K.get_value(model.optimizer.lr)
            K.set_value(model.optimizer.lr, lr * 0.1)
            print("lr changed to %d", lr * 0.1)
        return K.get_value(model.optimizer.lr)

    ### 调整学习速率
    reduce_lr = LearningRateScheduler(scheduler)

    borad = TensorBoard(log_dir='./logs', histogram_freq=0,
                        batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0,
                        embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                        update_freq='epoch')

    train_test_dataset = np.load("./dataset/train_test_dataset.npz")
    X_train = train_test_dataset['X_train']
    zeros=np.zeros((X_train.shape[0],10))
    X_train= np.c_[X_train,zeros]
    print(X_train.shape)
    Y_train_gate_change = train_test_dataset['Y_train_gate_change']
    X_test = train_test_dataset['X_test']
    zeros = np.zeros((X_test.shape[0],10))
    X_test= np.c_[X_test,zeros]
    print(X_test.shape)
    Y_test_gate_change = train_test_dataset['Y_test_gate_change']
    input_x=12
    input_y=13

    X_train = np.column_stack((X_train, X_train, X_train))
    X_test = np.column_stack((X_test, X_test, X_test))
    X_train = np.reshape(X_train, (X_train.shape[0], input_x, input_y, 3))
    X_test = np.reshape(X_test, (X_test.shape[0], input_x, input_y, 3))

    # labels = to_categorical(Y_train_last_stand,163)
    #
    # labels_test=to_categorical(Y_test_last_stand,163)

    from keras import layers
    from keras import models
    model = models.Sequential()
    model.add(layers.Conv2D(32, (2, 2), activation='relu',
                            input_shape=(input_x, input_y, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    # model.add(layers.Conv2D(32, (2, 2), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (2, 2), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train_gate_change, batch_size=512, nb_epoch=10,
              validation_data=(X_test, Y_test_gate_change),
              callbacks=[reduce_lr, borad])

    model.save('./model_output/cnn_gate_change.h5')
    print("训练完成，模型已保存在 model_output/cnn_gate_chane.h5 ")


if __name__ == "__main__":
    pass
