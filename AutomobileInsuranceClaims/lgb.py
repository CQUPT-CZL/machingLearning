import lightgbm as lgb
import pandas as pd
from hyperopt import hp, fmin, tpe

# D:\Data\AutomobileInsuranceClaims\


def read_data():
    train = pd.read_csv(r'D:\Data\AutomobileInsuranceClaims\train.csv')
    test = pd.read_csv(r'D:\Data\AutomobileInsuranceClaims\test.csv')
    return train, test

def preprocess(train, test):
    test = test.drop(['id'], axis=1)
    X = train.drop(['id', 'target'], axis=1)
    y = train.target
    return X, y, test

def addParams(params):
    params['metric'] = 'auc'

def model(X, y, test, params):
    # params = {
    #     'eta' : 0.008,
    #     'metric' : 'auc',
    #     'max_depth' : 10
    # }
    addParams(params)

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain)

    submit = pd.read_csv(r'D:\Data\AutomobileInsuranceClaims\submit.csv')
    submit['target'] = model.predict(test)
    submit.to_csv('submit_lgb.csv', index=False)

def para_hyperopt(train):
    def hyperopt_objective(params):
        res = lgb.cv(params,
                     train,
                     nfold=5,
                     shuffle=True,
                     metrics='auc',
                     early_stopping_rounds=15
                     )
        return min(res['auc-mean'])

    params_space = {
        'learning_rate': hp.uniform('learning_rate', 2e-4, 1e-2),
        'num_boost_round': hp.randint('n_estimators', 1800, 3000),
        'max_depth' : hp.randint('max_depth', 15, 40),
    }

    params_best = fmin(hyperopt_objective,
                       space=params_space,
                       algo=tpe.suggest,
                       max_evals=40
                       )

    return params_best

if __name__ == '__main__':

    train, test = read_data()

    X, y, test = preprocess(train, test)

    best_params = para_hyperopt(lgb.Dataset(X, label=y))

    model(X, y, test, best_params)
