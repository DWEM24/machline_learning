import sys

sys.path.append("..")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import pandas as pd

from data import get_data


class logistic_reg():
    def __init__(self, penalty='none', solver='lbfgs', fit_intercept=True):
        self.penalty = penalty
        self.solver = solver
        self.fit_intercept = fit_intercept

    def train(self, X_train, X_test, Y_train, Y_test, *args, **kwargs):
        if kwargs['penalty'] == 'l1':
            print("using:{}".format(kwargs['penalty']))
            model = LogisticRegression(penalty=kwargs['penalty'], solver=kwargs['solver'], fit_intercept= self.fit_intercept )
            model.fit(X_train, Y_train)
            print(model.coef_)
            e_train = metrics.mean_squared_error(Y_train, model.predict(X_train))
            e_test = metrics.mean_squared_error(Y_test, model.predict(X_test))
            print("训练集MSE：{}".format(e_train))
            print("测试集MSE：{}".format(e_test))

        else:
            print("using:{}".format(kwargs['penalty']))
            model = LogisticRegression(penalty=kwargs['penalty'],
                                       fit_intercept=self.fit_intercept,
                                       class_weight=kwargs['class_weight'],
                                       random_state=1)
            model.fit(X_train, Y_train)
            print(model.coef_)
            e_train = metrics.mean_squared_error(Y_train, model.predict(X_train))
            e_test = metrics.mean_squared_error(Y_test,model.predict(X_test))
            print("训练集MSE：{}".format(e_train))
            print("测试集MSE：{}".format(e_test))
            print("model score:{}".format(model.score(X_test,Y_test)))


if __name__ == '__main__':
    # 获取训练数据、测试数据
    X, Y = get_data.getdata()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    model = logistic_reg()
    model.train(X_train, X_test, Y_train, Y_test,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=1)  # 控制实验结果的随机性,方便对比实验结果。需要注意的是实验过程往往有多处使用了随机初始化，或者随机spilit





