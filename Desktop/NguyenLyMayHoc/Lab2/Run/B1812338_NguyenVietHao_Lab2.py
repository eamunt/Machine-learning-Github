import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#cau a
dataset = pd.read_csv("winequality-white.csv",delimiter=";")
#print(dataset)

#cau b
print("So phan tu:")
print(len(dataset)) #tập dữ liệu có 4898 phần tử

print("So nhan:")
print(np.unique(dataset['quality'])) # tập dữ liệu có 7 nhãn : [3 4 5 6 7 8 9]

X = dataset.drop(columns=['quality'])
#print(X)
y = dataset['quality']
#print(y)

#câu c
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=100)

#câu d  Xây dựng mô hình cây quyết định
model = DecisionTreeClassifier(criterion="entropy", random_state = 100)
model.fit(X_train, y_train)

#câu e
#độ chính xác tổng thể
y_pred = model.predict(X_test)
print("\nAccuracy :",accuracy_score(y_test,y_pred)*100)
#Accuracy : 61.73469387755102

#độ chính xác cho từng phân lớp
lb = np.unique(dataset['quality'])
print("\nConfusion matrix:")
print("\n",confusion_matrix(y_test, y_pred, labels=lb))

#câu f
print("\nAccuracy 6 elements: ",accuracy_score(y_test[0:6],y_pred[0:6])*100)
#Accuracy 6 elements:  50.0
print("\n")
print(y_pred[0:6]) 
print(y_test[0:6])



######### Bài 2
print("\nBai 2:\n")

XX = [ [180, 15, 0],
       [167, 42, 1],
       [136, 35, 1],
       [174, 15, 0],
       [141, 28, 1]
     ] 
yy = [0, 1, 1, 0, 1]

#xây dựng mô hình.
model1 = DecisionTreeClassifier(criterion="entropy", random_state = 100, min_samples_leaf=2)
model1.fit(XX,yy)

#dự đoán phần tử mới.
pre = model1.predict([[135, 39, 1]])
print("Prediction :",pre)



