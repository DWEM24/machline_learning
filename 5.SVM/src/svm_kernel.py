from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts
'''
‘linear’:线性核函数
‘poly’：多项式核函数
‘rbf’：径像核函数/高斯核
‘sigmod’:sigmod核函数
‘precomputed’:核矩阵
'''
# import our data 导入数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# split our data into train and test data 将数据分为训练集和测试集
X_train,X_test,y_train,y_test = ts(X, y, test_size=0.2)
print(y_test)
# select different type of kernel function and compare the score
# 选择不同的核函数并比较得分

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
print("The score of poly is : %f"%score_poly)
