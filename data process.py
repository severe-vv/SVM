import numpy as np
import re
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split # 分割数据模块

with open('faceDR','r') as f:
    for line in f.readlines():
        print(line)
        
label = []
num_delete = []
num = np.zeros([4,5])    #记录每个特征数量
with open('faceDR','r') as f:
    for line in f.readlines():
        m = re.findall(r' (\w+)',line) 
        if len(m) >2:
            m = m[0:5]
            m[0] = int(m[0])
            #判断性别
            if m[1] == 'female':
                m[1]= 0
                num[0][0] = num[0][0]+1
            elif m[1] == 'male':
                m[1] = 1
                num[0][1] = num[0][1]+1
            #判断年龄
            if m[2] == 'child':
                m[2]= 0
                num[1][0] = num[1][0]+1
            elif m[2] == 'teen':
                m[2] = 1
                num[1][1] = num[1][1]+1
            elif m[2] == 'adult':
                m[2] = 2
                num[1][2] = num[1][2]+1
            elif m[2] == 'senior':
                m[2] = 3
                num[1][3] = num[1][3]+1
            #判断肤色
            if m[3] == 'white':
                m[3]= 0
                num[2][0] = num[2][0]+1
            elif m[3] == 'black':
                m[3] = 1
                num[2][1] = num[2][1]+1
            elif m[3] == 'hispanic':
                m[3] = 2
                num[2][2] = num[2][2]+1
            elif m[3] == 'asian':
                m[3] = 3
                num[2][3] = num[2][3]+1
            elif m[3] == 'other':
                m[3] = 4
                num[2][4] = num[2][4]+1
            #判断表情    
            if m[4] == 'smiling':
                m[4]= 0
                num[3][0] = num[3][0]+1
            elif m[4] == 'funny':
                m[4] = 1
                num[3][1] = num[3][1]+1
            elif m[4] == 'serious':
                m[4] = 2
                num[3][2] = num[3][2]+1
            
            label.append(m)
        else :
            num_delete.append(int(m[0]))
num
label
import pandas as pd
num_delete
data = pd.read_csv('faceR',header =None,sep='\s+' ,engine = 'python')
data[0:10]
#删除无效数据
idx =[]
for i in num_delete:
    idx.append(np.where(data[0] == i)[0][0])
   # print(idx)
data = data.drop(idx)

x = data.iloc[:,1:].values.astype('float32')
#x = preprocessing.normalize(x, norm='l2')
x.shape
label = np.array(label)
y = label[:,1]
y.shape
x
#保存数据
np.save('test_data',x)
np.save('test_label',label)
import scipy.io as scio
# 读取matlab文件特征向量
d = scio.loadmat('ev.mat')
d
d['mean_face']

#knn
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=2)
#建立模型
knn = KNeighborsClassifier()
#训练模型
knn.fit(X_train, y_train)
#将准确率打印出
print(knn.score(X_test, y_test))

#支持向量机
clf1 = SVC(kernel='linear')
clf1.fit(x,y)
y_pred = clf1.predict(X_test)
miss_classified = (y_pred != y_test).sum()
print("MissClassified: ",miss_classified)
