from sklearn.model_selection import GridSearchCV
def load_datest(feature,dim):
    #
    x_r = np.load('raw_data.npy')
    label_r = np.load('label.npy')
    y_r = label_r[:,feature]
    x_s = np.load('test_raw_data.npy')
    label_s = np.load('test_label.npy')
    y_s = label_s[:,feature]
    X = np.vstack((x_r,x_s))
    Y = np.append(y_r,y_s)
    #数据归一化
    mean_px = X.mean().astype(np.float32)
    std_px = X.std().astype(np.float32)
    X = (X-mean_px)/std_px
    #划分数据
    if dim ==1 :
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2,test_size = 0.2)
    elif dim == 3:
        X = np.squeeze(X)
        X3 = np.zeros([X.shape[0],X.shape[1],X.shape[2],3])
        X3[:,:,:,0] = X
        X3[:,:,:,1] = X
        X3[:,:,:,2] = X
        X_train, X_test, Y_train, Y_test = train_test_split(X3, Y, random_state=2,test_size = 0.2)
    return X_train, X_test, Y_train, Y_test
 def cnn_model2():
    model = Sequential()

    model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu', input_shape = (128,128,1)))
    model.add(Convolution2D(filters = 32, kernel_size = (5,5),padding = 'Same',
                 activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
    model.add(Convolution2D(filters = 64, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation = "softmax"))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    return model
  model2 = cnn_model2()
dim =1 
X_train, X_test, Y_train, Y_test =load_datest(2,dim)
Y_test_h = to_categorical(Y_test)
Y_train_h = to_categorical(Y_train)
Y_test_h.shape
history = model2.fit(X_train,Y_train_h,epochs=5,validation_data=(X_test,Y_test_h), shuffle=False)
model2.save_weights('weight/cnn_weight2.h5')
model3 = cnn_model2()
model3.load_weights('weight/cnn_weight2.h5')

yp = model3.predict_classes(X_test)

yp_h = to_categorical(yp)  #预测结果热编码
#因为热编码后少了一个维度（没有预测到3，实际数据是有3的）所以增加一维
yp = np.zeros(Y_test_h.shape)
yp[:,0:3] = yp_h
plot_auc(Y_test_h,yp)

# cnn倒数第二层输出
dense1_layer_model = Model(inputs=model3.input,outputs=model3.layers[-2].output)
X_rtemp = dense1_layer_model.predict(X_train)
X_stemp = dense1_layer_model.predict(X_test)
np.save('cnn_output/X_rtemp2',X_rtemp)
np.save('cnn_output/X_stemp2',X_stemp)
np.save('cnn_output/Y_train2',Y_train)
np.save('cnn_output/Y_test2',Y_test)
np.save('cnn_output/X_train2',X_train)
np.save('cnn_output/X_test2',X_test)
