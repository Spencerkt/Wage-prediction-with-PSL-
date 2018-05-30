import pandas as pd  
import numpy as np
import sys

def show_csv(path):
    dataset_df = pd.read_csv(path)
    print('shape = ', dataset_df.shape)
    print()
    print(dataset_df.head(25))
    print('\n----------------------------------------------\n')
    print('occs => ', dataset_df.OCCUPATION.unique())
    print('sector => ', dataset_df.SECTOR.unique())
    print('education => ', np.sort(dataset_df.EDUCATION.unique()))
    print('race => ', dataset_df.RACE.unique())
    print('wages => ', np.unique([int(x) for x in dataset_df.WAGE.values]))
    print('experience => ', np.unique([int(x) for x in dataset_df.EXPERIENCE.values]))
    print('age => ', np.unique([int(x) for x in dataset_df.AGE.values]))





if __name__ == '__main__':

    train_path = './../TRAIN/dataset_TRAIN/tables/learningData.csv'
    test_path = './../TEST/dataset_TEST/tables/learningData.csv'

    if sys.argv[1] == 'train':
        show_csv(train_path)
    
    if sys.argv[1] == 'test':
        show_csv(test_path)