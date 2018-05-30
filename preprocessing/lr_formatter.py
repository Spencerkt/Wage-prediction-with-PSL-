import pandas as pd  
import numpy as np
import sys


def lr_format(path, train=True):
    df = pd.read_csv(path)

    df.SOUTH = pd.Series(np.where(df.SOUTH == 'yes', 1, 0))
    df.SEX = pd.Series(np.where(df.SEX == 'male', 1, 0))
    df.UNION = pd.Series(np.where(df.UNION == 'member', 1, 0))
    df.RACE = pd.Series(np.where(df.RACE == 'white', 1, 0))
    occs = {'Other': 0, 'Management': 1, 'Sales': 2, 'Clerical': 3, 'Service': 4, 'Professional': 5}
    df.OCCUPATION = df.OCCUPATION.map(occs)
    sects = {'Manufacturing': 0, 'Other': 1, 'Construction': 2}
    df.SECTOR = df.SECTOR.map(sects)
    df.MARR = pd.Series(np.where(df.MARR == 'Married', 1, 0))

    if train:
        bins = 18   # found to be the best
        df.WAGE = pd.cut(df.WAGE, bins, labels=list(range(bins)))
        df.to_csv('./wage_train_disc.csv', index=False)
    else:
        df = df.drop(['WAGE'], axis=1)
        df.to_csv('./wage_test_disc.csv', index=False)


    return 


def lr_target_format(target_path):
    df = pd.read_csv(target_path)
    df.to_csv('./wage_targets_dics.csv', index=False)



if __name__ == '__main__':

    train_path = './../TRAIN/dataset_TRAIN/tables/learningData.csv'
    test_path = './../TEST/dataset_TEST/tables/learningData.csv'
    target_path = './../SCORE/targets.csv'

    lr_format(train_path)
    # lr_format(test_path, train=False)

    # lr_target_format(target_path)
