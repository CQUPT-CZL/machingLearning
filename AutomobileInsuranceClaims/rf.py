import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    model = RandomForestClassifier(n_estimators=500, n_jobs= - 1)
    model.fit(X, y)

    submit = pd.read_csv(r'D:\Data\AutomobileInsuranceClaims\submit.csv')
    submit['target'] = model.predict(test)
    submit.to_csv('submit_rf.csv', index=False)


if __name__ == '__main__':

    train, test = read_data()

    X, y, test = preprocess(train, test)

    model(X, y, test)