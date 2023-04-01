from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data import get_data

# 读取数据reading data

X ,Y = get_data.get_data()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# 建模
LR = LinearRegression()
LR.fit(X_train, Y_train)

print(LR.score(X_train, Y_train))
print(LR.score(X_test, Y_test))











