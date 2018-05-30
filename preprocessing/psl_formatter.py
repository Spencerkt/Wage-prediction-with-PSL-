import pandas as pd 
from sklearn import preprocessing


def write_local_wage_file(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    lw_df = pd.concat([train, test])
    lw_df.to_csv('./LocalWage_obs.txt', sep='\t', index=False, header=None)
    return 



def write_demo_files(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train = train.drop(['WAGE'], axis=1)
    demo_df = pd.concat([train, test])

    isMale = demo_df[['d3mIndex', 'SEX']]
    isMale = isMale.loc[isMale.SEX == 'male']
    isMale['true'] = 1.0 
    isMale = isMale.drop(['SEX'], axis=1)
    isMale.to_csv('./isMale_obs.txt', sep='\t', index=False, header=None)

    occs = demo_df[['d3mIndex', 'OCCUPATION']]
    isManagment = occs.loc[occs.OCCUPATION == 'Management'] 
    isProfessional = occs.loc[occs.OCCUPATION == 'Professional']

    isManagment['true'] = 1.0
    isManagment = isManagment.drop(['OCCUPATION'], axis=1)
    isManagment.to_csv('./isManagment_obs.txt', sep='\t', index=False, header=None)

    isProfessional['true'] = 1.0 
    isProfessional = isProfessional.drop(['OCCUPATION'], axis=1)
    isProfessional.to_csv('./isProfessional_obs.txt', sep='\t', index=False, header=None)

    isUnion = demo_df[['d3mIndex', 'UNION']]
    isUnion = isUnion.loc[isUnion.UNION == 'member']
    isUnion['true'] = 1.0
    isUnion = isUnion.drop(['UNION'], axis=1)
    isUnion.to_csv('./isUnion_obs.txt', sep='\t', index=False, header=None)

    return 


def write_wage_files(demo_path, target_path):
    demo_df = pd.read_csv(demo_path)

    wage_truth = pd.read_csv(target_path)
    wage_obs = demo_df[['d3mIndex', 'WAGE']]
    
    truth_ind = wage_truth.d3mIndex.values
    obs_ind = wage_obs.d3mIndex.values

    all_wages = pd.concat([wage_obs, wage_truth])
    scaler = preprocessing.MinMaxScaler()
    all_wages[['WAGE']] = scaler.fit_transform(all_wages[['WAGE']])


    scaled_wage_obs = all_wages.iloc[obs_ind]
    scaled_wage_obs.to_csv('./Wage_obs.txt', sep='\t', index=False, header=None)

    scaled_wage_truth = all_wages.iloc[truth_ind]
    scaled_wage_truth.to_csv('./Wage_truth.txt', sep='\t', index=False, header=None)
    scaled_wage_truth.drop(['WAGE'], axis=1).to_csv('./Wage_targets.txt', sep='\t', index=False, header=None)
    return




if __name__ == '__main__':

    lw_test_path = './local_wage_test.csv'
    lw_train_path = './local_wage_train.csv'

    demographic_train_path = './../TRAIN/dataset_TRAIN/tables/learningData.csv'
    demographic_test_path = './../TEST/dataset_TEST/tables/learningData.csv'
    wage_truths_path = './../SCORE/targets.csv'

    write_local_wage_file(lw_train_path, lw_test_path)
    write_demo_files(demographic_train_path, demographic_test_path)
    write_wage_files(demographic_train_path, wage_truths_path)
    

