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
    sub = pd.read_csv(path + '提交示例.csv')
    return train, test, sub


def model_train(X, y, val_X, val_y):
    # 设置XGBoost参数
    params = {
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'objective': 'binary',
        'metric': 'auc',
        'min_child_weight': 5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'seed': 2022,
        'n_jobs': -1
    }

    dtrain = lgb.Dataset(X, label=y)
    val = lgb.Dataset(val_X, label=val_y)
    #  训练模型
    model = lgb.train(params, dtrain, num_boost_round=1500, verbose_eval=100, early_stopping_rounds=200,
                      valid_sets=[dtrain, val])

    return model


def train_01(x):
    if x < 0.28:
        return 0
    return 1


if __name__ == '__main__':
    print('start')
    train, test, sub = read_data()
    print(train.head())


    X = train.drop(['target', 'udmap', 'common_ts', 'common_ts_datatime'], axis=1)
    y = train.target

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

    model = model_train(X_train, y_train, X_test, y_test)
    predictions = model.predict(X_test)
    predictions01 = np.round(predictions)
    score = f1_score(y_true=y_test, y_pred=predictions01, average='macro')
    print(score)

    # pd.DataFrame({
    #     'uuid': test['uuid'],
    #     'target': model.predict(test.drop(['udmap', 'common_ts', 'common_ts_datatime'], axis=1))
    # }).to_csv('submit.csv', index=None)

    sub['target'] = model.predict(test.drop(['udmap', 'common_ts', 'common_ts_datatime'], axis=1))
    sub['target'] = sub['target'].apply(lambda x : train_01(x))

    sub.to_csv('submit.csv', index = False)

