from sklearn import datasets
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


def getdata():
    loaded_data = datasets.load_breast_cancer()
    bc_x = pd.DataFrame(loaded_data.data)
    bc_y = pd.Series(loaded_data.target)
    bc_data = pd.concat([bc_x, bc_y], axis=1)
    bc_data.columns = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension",
        "radius error",
        "texture error",
        "perimeter error",
        "area error",
        "smoothness error",
        "compactness error",
        "concavity error",
        "concave points error",
        "symmetry error",
        "fractal dimension error",
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
        "worst smoothness",
        "worst compactness",
        "worst concavity",
        "worst concave points",
        "worst symmetry",
        "worst fractal dimension",
        "target"
    ]
    X = bc_data.iloc[:, :-1]
    Y = bc_data['target']

    return X, Y

# if __name__ == "__main__":
#     X, Y = getData.get_data()
