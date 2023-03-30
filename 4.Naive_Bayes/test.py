# 写一个示例，使用sklearn中的朴素贝叶斯分类器

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

# 加载数据
iris = datasets.load_iris()
features = iris.data
target = iris.target

# 创建高斯朴素贝叶斯对象
classifer = GaussianNB()

# 训练模型
model = classifer.fit(features, target)

# 创建新观察值
new_observation = [[ 4,  4,  4,  0.4]]

# 预测观察值的分类
model.predict(new_observation)

# 查看预测的概率
model.predict_proba(new_observation)

# 查看模型的准确度
model.score(features, target)

# 查看模型的类别
model.classes_

# 查看模型的特征
model.theta_

# 查看模型的方差
model.sigma_

# 查看模型的先验概率
model.class_prior_

# 查看模型的特征数量
model.n_features_

# 查看模型的类别数量
model.n_classes_

# 查看模型的训练样本数量





