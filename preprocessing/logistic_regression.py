import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error


def lr_model(x_train, y_train, x_test, y_test):
    LR = LogisticRegression()
    LR.fit(x_train, y_train)

    y_test_predictions = LR.predict(x_test)
    print('MSE test => ', mean_squared_error(y_test.WAGE.values, y_test_predictions))
    x_test['PRED'] = y_test_predictions
    x_test[['d3mIndex', 'PRED']].to_csv('./local_wage_test.csv', index=False)

    y_train_predictions = LR.predict(x_train)
    print('MSE train => ', mean_squared_error(y_train.values, y_train_predictions))
    x_train['PRED'] = y_train_predictions
    x_train[['d3mIndex', 'PRED']].to_csv('./local_wage_train.csv', index=False)

    return 






if __name__ == '__main__':

    train = pd.read_csv('./wage_train_disc.csv')
    test_x = pd.read_csv('./wage_test_disc.csv')
    test_y = pd.read_csv('./wage_targets_disc.csv')

    train_y = train.WAGE
    train_x = train.drop(['WAGE'], axis=1)

    lr_model(train_x, train_y, test_x, test_y)
