import sys
sys.path.append("..")

from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from lib import get_data as gd

# import our data 导入数据
x_train, y_train = gd.read_data("../data/train_data")
x_test, y_test = gd.read_data("../data/test_data")
print(type(x_train))

# 带松弛变量的SVM（SVC）
model = svm.SVC(C=1.0, kernel='rbf', gamma=0.1, decision_function_shape='ovr')
model.fit(x_train, y_train.ravel()) # 将多维数组转化为一维数组
print(model.support_vectors_)  # 打印支持向量的点
print(model.support_)
print(len(model.support_))

# 打印测试集的准确率
print(model.score(x_test, y_test))

# 打印训练集的准确率
print(model.score(x_train, y_train))

# 交叉验证
# 1. 选择模型
model = svm.SVC()
# 2. 选择参数
parameters = {'kernel':('linear', 'rbf'), 'C':np.logspace(-3, 3, 7)}
# 3. 选择评估指标
scoring = 'accuracy'
# 4. 选择交叉验证策略
clf = GridSearchCV(model, parameters, scoring=scoring, cv=5)
# 5. 训练模型
clf.fit(x_train, y_train.ravel()) # 训练集被划分为5份，每次训练4份，验证1份
# 6. 打印最佳参数
print('最佳参数:', clf.best_params_)
# 7. 打印最佳模型
print('最佳模型:', clf.best_estimator_)
# 8. 打印最佳准确率
print('最佳准确率:', clf.best_score_)






