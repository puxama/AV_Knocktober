__author__ = 'puxama'
import pandas as pd
import numpy as np
from random import seed
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn import metrics

seed(4321)





ds_camp = pd.read_csv('Files/ds_camp_end.csv')

print ds_camp.columns.values

predictors = [var for var in ds_camp if var not in ['Patient_ID', 'Health_Camp_ID', 'Registration_Date',
                                                    'Camp_Start_Date', 'Camp_End_Date', 'First_Interaction',
                                                    'Target', 'Type', 'Var3']]


ds_train = ds_camp.query("Type == 0")
ds_test = ds_camp.query("Type == 1")

print predictors



xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=9,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

tunning = [10]

rf = RandomForestClassifier(n_estimators=1000, criterion='entropy',
                                min_samples_leaf=30, bootstrap=True, random_state=1234, min_samples_split=30)

for tun in tunning:

    model = rf

    model_list = []
    ds_train['Test'] = -1

    cv = np.random.randint(0, 5, ds_train.shape[0])
    auc_final = pd.DataFrame({"Patient_ID": [], "Health_Camp_ID": [],
                           "Target": [], "Test":[]})
    for fold in xrange(0, 5):
        ind_train = cv != fold
        ind_val = cv == fold
        model.fit(ds_train[ind_train][predictors], ds_train[ind_train]['Target'])
        model_list.append(model)
        ds_temp = ds_train[ind_val][["Patient_ID", "Health_Camp_ID", "Target", "Test"]]
        pred = model.predict_proba(ds_train[ind_val][predictors])[:, 1]
        ds_temp["Test"] = pred
        auc_final = pd.concat([auc_final,ds_temp])
        print "AUC Score (Train): %f" % metrics.roc_auc_score(ds_train[ind_val]['Target'], pred)

        #names_var = list(ds_train[predictors].columns.values)
        #importances = rf.feature_importances_
        #indices = np.argsort(importances)[::-1]

        #print('Feature ranking')
        #for f in range(ds_train[predictors].shape[1]):
        #    print("%d. feature %d %s (%f)" % (f+1, indices[f], names_var[indices[f]], importances[indices[f]]))


    del(ds_temp)
    print "min sample leaf: %f" % tun
    print "AUC Score (Final): %f" % metrics.roc_auc_score(auc_final['Target'], auc_final['Test'])
    auc_final.to_csv('rf3.csv',index=False)

    #modelfit(rf, ds_train[ind_train], ds_train[ind_val], 'Target', predictors)
    #print "train mean ", ds_train[ind_train]['Target'].mean(), ds_train[ind_train].shape
    #print "test mean ", ds_train[ind_val]['Target'].mean(), ds_train[ind_val].shape


num_model = 1
name_model = []
for model in model_list:
    name_model.append('model'+str(num_model))
    ds_test['model'+str(num_model)] = model.predict_proba(ds_test[predictors])[:, 1]
    num_model += 1

ds_test['Outcome'] = ds_test[name_model].mean(axis=1)

out_df = pd.DataFrame({"Patient_ID": ds_test['Patient_ID'], "Health_Camp_ID": ds_test['Health_Camp_ID'],
                       "Outcome": ds_test['Outcome']})
out_df.to_csv("sub_rf3.csv", index=False)

