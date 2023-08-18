import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score

def read_data():
    path = 'D:\Code\Python\PythonLearning\用户新增预测挑战赛\input/'
    train = pd.read_csv(path + 'new_train.csv')
    test = pd.read_csv(path + 'new_test.csv')
    return train, test

def model_train(X, y):
    # 设置XGBoost参数
    params = {
        'eta': 0.01,  # 学习率
    }

    dtrain = xgb.DMatrix(X, label=y)

    model = xgb.train(params, dtrain, num_boost_round=100)

    return model

if __name__ == '__main__':

    train, test = read_data()

    X = train.drop(['target', 'udmap', 'common_ts', 'common_ts_datatime'], axis=1)
    y = train['target']


    print(y.shape, X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

    model = model_train(X_train, y_train)

    predictions = model.predict(xgb.DMatrix(X_test))
    predictions01 = np.round(predictions)
    score = f1_score(y_true=y_test, y_pred=predictions01, average='macro')
    print(score)

    pd.DataFrame({
        'uuid': test['uuid'],
        'target': np.round(model.predict(xgb.DMatrix(test.drop(['udmap', 'common_ts', 'common_ts_datatime'], axis=1))))
    }).to_csv('submit.csv', index=None)









