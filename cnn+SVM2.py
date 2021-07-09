from keras.models import Sequential,Model
from keras.layers import Convolution2D,MaxPool2D,Flatten,Dense,Dropout,Lambda
from keras.callbacks import TensorBoard
import numpy as np
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import models

#加载数据
X_R = np.load('raw_data.npy')
label_R = np.load('label.npy')
Y_R = label_R[:,1]
X_S = np.load('test_raw_data.npy')
label_S = np.load('test_label.npy')
Y_S = label_S[:,1]

X = np.vstack((X_R,X_S))
Y = np.append(Y_R,Y_S)

#数据归一化
mean_px = X.mean().astype(np.float32)
std_px = X.std().astype(np.float32)
X = (X-mean_px)/std_px

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=2,test_size = 0.2)

X_train.shape

def model():
    model=Sequential([
    Convolution2D(32,3,3,input_shape=(128,128,1),activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Convolution2D(32,3,3,input_shape=(128,128,1),activation='relu'),
    MaxPool2D(pool_size=(2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
   ])
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model
  #训练保存模型
#model = model()
#model.fit(X_train,y_train,epochs=1,validation_data=(X_test,y_test))
#model.save('cnn82.h5')
#model.save_weights('cnn82_weight.h5')
model2 = model()

model2.load_weights('cnn85_weight.h5')

dense1_layer_model = Model(inputs=model2.input,outputs=model2.layers[-2].output)

from sklearn.svm import SVC
import numpy as np
clf=SVC()
X_trian2 = dense1_layer_model.predict(X_train)
X_test2 = dense1_layer_model.predict(X_test)
clf.fit(X_trian2,y_train)
y_p = clf.predict(X_test2)
acc = np.sum(y_p == y_test)/y_test.shape[0]
acc
y_p2 = model2.predict_classes(X_test)
acc2 = np.sum(np.squeeze(y_p2) == y_test)/y_test.shape[0]
acc2
