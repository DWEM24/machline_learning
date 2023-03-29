import pandas as pd

import statsmodels.api as sm


def get_data():
    from sklearn.datasets import fetch_california_housing as fch
    # 导入加利福尼亚数据集
    fch = fch(as_frame=False)  # 返回值类型pd
    cfx = pd.DataFrame(fch.data)  # 使用panda解析
    # print(type(cfx))
    cfy = pd.Series(fch.target)
    cfdata = pd.concat([cfx, cfy], axis=1)
    cfdata.columns = ['medinc', 'houseage', 'averooms', 'avebedrms', 'population',
                      'aveoccup', 'latitude', 'longitude', 'price']
    # 添加截距项，调整模型预测水平，通常是哟哦那个线性回归是会添加一个截距项
    cfdata = sm.add_constant(cfdata)

    X = cfdata.iloc[:, :-1]
    Y = cfdata['price']

    return X, Y
