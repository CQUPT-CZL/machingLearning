import pandas as pd
import xgboost as xgb
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

def model(X, y, test):
    params = {
        'eta' : 0.0032730172253505716,
        'eval_metric' : 'auc',
        'max_depth' : 10
    }

    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=1546)
    submit = pd.read_csv(r'D:\Data\AutomobileInsuranceClaims\submit.csv')
    submit['target'] = model.predict(xgb.DMatrix(test))
    submit.to_csv('submit_xgb.csv', index=False)

def para_hyperopt(train):
    def hyperopt_objective(params):
        res = xgb.cv(params,
                     train,
                     num_boost_round= params['num_boost_round'],
                     nfold=5,
                     shuffle=True,
                     metrics='auc',
                     early_stopping_rounds=10
                     )
        return min(res['train-auc-mean'])

    params_space = {
        'learning_rate': hp.uniform('learning_rate', 2e-4, 1e-2),
        'num_boost_round': hp.randint('n_estimators', 1500, 2000)
    }
    print('start')
    params_best = fmin(hyperopt_objective,
                       space=params_space,
                       algo=tpe.suggest,
                       max_evals=30
                       )

    print(params_best)
    return params_best

if __name__ == '__main__':

    train, test = read_data()

    X, y, test = preprocess(train, test)

    # best_params = para_hyperopt(xgb.DMatrix(X, label=y))

    model(X, y, test)