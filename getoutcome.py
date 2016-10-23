__author__ = 'puxama'

import pandas as pd

ds_test = pd.read_csv('Files/Test_D7W1juQ.csv')
ds_train = pd.read_csv('Files/Train.csv')
ds_patient = pd.read_csv('Files/Patient_Profile.csv')
ds_camp = pd.read_csv('Files/Health_Camp_Detail.csv')
ds_first_camp = pd.read_csv('Files/First_Health_Camp_Attended.csv')
ds_second_camp = pd.read_csv('Files/Second_Health_Camp_Attended.csv')
ds_third_camp = pd.read_csv('Files/Third_Health_Camp_Attended.csv')

ds_list = [ds_train, ds_patient, ds_camp, ds_first_camp, ds_second_camp, ds_third_camp]


train_end = pd.merge(ds_train, ds_patient, how='left', on='Patient_ID')
train_end = pd.merge(train_end, ds_camp, how='left', on='Health_Camp_ID')
train_end = pd.merge(train_end, ds_first_camp, how='left', on=['Patient_ID', 'Health_Camp_ID'])
train_end = pd.merge(train_end, ds_second_camp, how='left', on=['Patient_ID', 'Health_Camp_ID'])
train_end = pd.merge(train_end, ds_third_camp, how='left', on=['Patient_ID', 'Health_Camp_ID'])

test_end = pd.merge(ds_test, ds_patient, how='left', on='Patient_ID')
test_end = pd.merge(test_end, ds_camp, how='left', on='Health_Camp_ID')
test_end['Target'] = 0



train_end['Target'] = 0
train_end.loc[(train_end['Health_Score'].notnull()) |
              (train_end['Health Score'].notnull()) |
              ((train_end['Number_of_stall_visited'].notnull()) &
               (train_end['Number_of_stall_visited'] >= 1)), 'Target'] = 1

train_end.drop(['Health_Score', 'Health Score',
                'Number_of_stall_visited', 'Last_Stall_Visited_Number', 'Donation'], axis=1, inplace=True)


print train_end.columns.values
print train_end.shape
train_end.to_csv('Files/Train_end.csv', index=False)

print test_end.columns.values
print test_end.shape
test_end.to_csv('Files/Test_end.csv', index=False)


