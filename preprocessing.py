__author__ = 'puxama'

import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import numpy as np

ds_train = pd.read_csv('Files/Train_end.csv', na_values='None')
ds_test = pd.read_csv('Files/Test_end.csv', na_values='None')

#Eliminando missing values RegistrationDate

ds_train = ds_train.loc[ds_train['Registration_Date'].notnull()]

#concatenando 2 datasets
ds_train['Type'] = 0
ds_test['Type'] = 1

ds_camp = pd.concat([ds_train, ds_test])

#Dando formato a las variables de tipo date
ds_camp['Registration_Date'] = pd.to_datetime(ds_camp['Registration_Date'], format='%d-%b-%y')
ds_camp['First_Interaction'] = pd.to_datetime(ds_camp['First_Interaction'], format='%d-%b-%y')
ds_camp['Camp_Start_Date'] = pd.to_datetime(ds_camp['Camp_Start_Date'], format='%d-%b-%y')
ds_camp['Camp_End_Date'] = pd.to_datetime(ds_camp['Camp_End_Date'], format='%d-%b-%y')


ds_camp.sort_values(['Registration_Date'], inplace=True)

ds_camp['Num_assistances'] = ds_camp.groupby(['Patient_ID'])['Target'].cumsum()-1
ds_camp['Num_of_registration'] = ds_camp.groupby(['Patient_ID']).cumcount()+1
ds_camp.loc[ds_camp['Num_assistances'] < 0, 'Num_assistances'] = 0
ds_camp['Porc_assistances'] = ds_camp['Num_assistances']*1.0/ds_camp['Num_of_registration']


#creando variables con fechas

ds_camp['Days_bt_Registration_CampStart'] = \
    (ds_camp['Camp_Start_Date'] - ds_camp['Registration_Date']).astype('timedelta64[D]')
ds_camp['Camp_duration'] = \
    (ds_camp['Camp_End_Date'] - ds_camp['Camp_Start_Date']).astype('timedelta64[D]')
ds_camp['Left_time_togo'] = \
    (ds_camp['Camp_End_Date'] - ds_camp['Registration_Date']).astype('timedelta64[D]')
ds_camp['Time_knowledge_camp'] = \
    (ds_camp['Registration_Date'] - ds_camp['First_Interaction']).astype('timedelta64[D]')

#imputing missing values

ds_camp['City_Type'].fillna('Z', inplace=True)
ds_camp['Employer_Category'].fillna('NoReg', inplace=True)

ds_camp['Income'].fillna(-999, inplace=True)
ds_camp['Income'] = ds_camp['Income'].astype(float)

ds_camp['Education_Score'].fillna(-999, inplace=True)
ds_camp['Education_Score'] = ds_camp['Education_Score'].astype(float)

ds_camp['Age'].fillna(-999, inplace=True)
ds_camp['Age'] = ds_camp['Age'].astype(float)

predictors = [var for var in ds_camp if ds_camp not in []]

list_var = ['Var1', 'Var2', 'Var3', 'Var4', 'Var5']
list_sn = ['Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared']

ds_camp['VarZero'] = 5 - ds_camp[list_var].astype(bool).sum(axis=1)
ds_camp['numSN'] = ds_camp[list_sn].astype(bool).sum(axis=1)

ds_camp['Left_time_togo_porc'] = ds_camp['Left_time_togo']*1.0/ds_camp['Camp_duration']
ds_camp.loc[ds_camp['Days_bt_Registration_CampStart'] > 0, 'Left_time_togo_porc'] = 1

ds_camp['Month_Camp_Start'] = ds_camp['Camp_Start_Date'].dt.month
ds_camp['Year_Camp_Start'] = ds_camp['Camp_Start_Date'].dt.year

ds_camp['Month_Camp_End'] = ds_camp['Camp_End_Date'].dt.month
ds_camp['Year_Camp_End'] = ds_camp['Camp_End_Date'].dt.year

ds_camp['Month_Registration'] = ds_camp['Registration_Date'].dt.month
ds_camp['Year_Registration'] = ds_camp['Registration_Date'].dt.year

ds_camp['NoIncome'] = 1
ds_camp.loc[ds_camp['Income']!= -999, 'NoIncome'] = 0

ds_camp['NoEducation'] = 1
ds_camp.loc[ds_camp['NoEducation']!= -999, 'NoEducation'] = 0

ds_camp['NoAge'] = 1
ds_camp.loc[ds_camp['NoAge']!= -999, 'NoAge'] = 0

list_noinformation = ['NoIncome', 'NoEducation', 'NoAge']

ds_camp['NoInformation'] = ds_camp[list_noinformation].astype(bool).sum(axis=1)

ds_camp.drop(list_noinformation, axis=1, inplace=True)

for var in ds_camp:
    if ds_camp[var].dtype == object:
        lbl = LabelEncoder()
        lbl.fit(np.array(ds_camp[var]))
        ds_camp[var] = lbl.transform(ds_camp[var])

print ds_camp.columns.values

ds_camp.to_csv('Files/ds_camp_end.csv', index=False)

