import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

def read_data():
    path = 'D:\Code\Python\PythonLearning\用户新增预测挑战赛\input/'
    train = pd.read_csv(path + 'new_train.csv')
    test = pd.read_csv(path + 'new_test.csv')
    submit = pd.read_csv(path + '提交示例.csv')
    return train, test, submit


def model_train(X, y):
    # 设置XGBoost参数
    params = {
        'learning_rate': 0.01,
    }

    dtrain = lgb.Dataset(X, label=y)
    #  训练模型
    model = lgb.train(params, dtrain)

    return model


def train_01(x):
    if x < 0.5:
        return 0
    return 1


if __name__ == '__main__':
    print('start')
    train, test, submit = read_data()
    print(train.head())

    train = train.astype(int)
    test = test.astype(int)

    X = train.drop(['uuid', 'target'], axis=1)
    y = train.target

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

    model = model_train(X_train, y_train)
    predictions = model.predict(X_test)
    predictions01 = np.round(predictions)
    score = f1_score(y_true=y_test, y_pred=predictions01, average='macro')
    print(score)

    submit['target'] = model.predict(test.drop(['uuid'], axis=1))
    submit['target'] = submit['target'].apply(lambda x: train_01(x))
    submit.to_csv('submit.csv', index=False)
