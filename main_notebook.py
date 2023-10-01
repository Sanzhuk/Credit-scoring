#!/usr/bin/env python
# coding: utf-8

# ## Установка библиотек для чтения pq-файлов

# In[ ]:


get_ipython().system('pip install fastparquet ')


# In[ ]:


get_ipython().system('pip install pyarrow')


# In[9]:


get_ipython().system('pip install xgboost')


# In[2]:


import pandas as pd


# # Проект 03. Кредитный скоринг

# **Кредитный скоринг** является важной банковской задачей. 
# 
# Стандартный подход заключается в построении классических моделей машинного обучения, таких как логистическая регрессия и градиентный бустинг на табличных данных, в том числе с использованием *агрегирования* из некоторых последовательных данных, таких как истории транзакций клиентов. 
# 

# ### Описание полей:
# - id - identifier of the application
# - <span style = 'color : red'> **flag** </span> - target (целевая переменная)
# - pre_since_opened - days from credit opening date to data collection date
# - pre_since_confirmed - days from credit information confirmation date till data collection date
# - pre_pterm - planned number of days from credit opening date to closing date
# - pre_fterm - actual number of days from credit opening date to closing date
# - pre_till_pclose - planned number of days from data collection date until loan closing date
# - pre_till_fclose - actual number of days from data collection date until loan closing date
# - pre_loans_credit_limit - credit limit
# - pre_loans_next_pay_summ - amount of the next loan payment
# - pre_loans_outstanding - outstanding loan amount
# - pre_loans_total_overdue - current overdue amount
# - pre_loans_max_overdue_sum - maximum overdue amount
# - pre_loans_credit_cost_rate - total cost of credit
# - pre_loans5 - number of delinquencies of up to 5 days
# - pre_loans530 - number of delinquencies from 5 to 30 days
# - pre_loans3060 - number of delinquencies from 30 to 60 days
# - pre_loans6090 - number of delinquencies from 60 to 90 days
# - pre_loans90 - number of delinquencies of more than 90 days
# - is_zero_loans_5 - flag: no delinquencies of up to 5 days
# - is_zero_loans_530 - flag: no delinquencies of 5 to 30 days
# - is_zero_loans_3060 - flag: no delinquencies of 30 to 60 days
# - is_zero_loans_6090 - flag: no delinquencies of 60 to 90 days
# - is_zero_loans90 - flag: no delinquencies of more than 90 days
# - pre_util - ratio of outstanding loan amount to credit limit
# - pre_over2limit - ratio of currently overdue debt to credit limit
# - pre_maxover2limit - ratio of maximum overdue debt to credit limit
# - is_zero_util - flag: ratio of outstanding loan amount to credit limit equals 0
# - is_zero_over2limit - flag: ratio of current overdue debt to credit limit equals 0
# - is_zero_maxover2limit - flag: ratio of maximum overdue debt to credit limit equals 0
# - **<span style = 'color : blue'> enc_paym_{0…n} </span>** - monthly payment statuses of the last n months
# - enc_loans_account_holder_type - type of relation to the loan
# - enc_loans_credit_status - credit status
# - enc_loans_account_cur - currency of the loan
# - enc_loans_credit_type - credit type
# - pclose_flag - flag: planned number of days from opening date to closing date of the loan
# - fclose_flag - flag: actual number of days from credit opening date to closing date undefined

# # Submission

# Вам необходимо проскорить файл **'P03_test.pq'** и записать 2 столбца:
# - id
# - prediction
# 
# 

# # Open files

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
from numpy import loadtxt
from numpy import sort
from sklearn.feature_selection import SelectFromModel
import pickle
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from xgboost import plot_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold


# In[2]:


def print_metrics(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1-score: {f1_score(y_test, y_pred)}')
    print(f'AUC: {roc_auc_score(y_test, y_pred)}')


# In[3]:


df_train = pd.read_parquet('P03_train.pq')
df_test = pd.read_parquet('P03_test.pq')


# In[4]:


df_train


# In[10]:


loaded_model = pickle.load(open('lgbm_cb_xgb_ensemble.pickle', "rb"))

X = pd.read_csv('balanced_train_X.csv').drop('Unnamed: 0', axis = 1)
y = pd.read_csv('balanced_train_y.csv').drop('Unnamed: 0', axis = 1)


# In[15]:


loaded_model.fit(X, y)


# In[16]:


predictions = loaded_model.predict(df_test.drop('id', axis = 'columns'))


# In[ ]:


X = df_train.drop('id')


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[19]:


predictions_df = pd.DataFrame({
    'id' : df_test['id'],
    'prediction' : predictions
})


# In[25]:


df_test


# In[24]:


predictions_df['prediction'].value_counts()


# # Data preprocessing

# In[9]:


df_train = df_train.drop(columns = ['id'])


# In[10]:


df_train = df_train.drop_duplicates()


# In[11]:


df_train.isna().any().any()


# In[14]:


df_train['pre_loans_total_overdue'].value_counts()


# In[81]:


correlation_with_target = df_train.corr()['flag']
sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

sorted_correlation


# In[10]:


correlation_matrix = df_train.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(30, 30))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[15]:


X = df_train.drop(columns = ['flag'])
y = df_train['flag']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
mtrain_X, mtest_X, mtrain_y, mtest_y = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.25)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)


# In[15]:


y_train.value_counts()


# In[16]:


97201/3582


# In[84]:


bestfeatures = SelectKBest(score_func=chi2, k=59)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(59,'Score'))  #print 10 best features


# In[86]:


threshold = 10

selected_features = featureScores[featureScores['Score'] >= threshold]
selected_feature_names = selected_features['Specs'].tolist()
df_train_selected = df_train[selected_feature_names]


# In[91]:


selected_X = df_train_selected
selected_y = df_train['flag']

selected_X_train, selected_X_test, selected_y_train, selected_y_test = train_test_split(selected_X, selected_y, test_size = 0.3, random_state = 42)


# In[92]:


xgb_model = xgb.XGBClassifier(scale_pos_weight = 27)
xgb_model.fit(selected_X_train, selected_y_train)


# In[93]:


y_preds = xgb_model.predict(selected_X_test)

print_metrics(selected_y_test, y_preds)


# In[95]:


over = SMOTE(sampling_strategy=0.1)

new_x, new_y = over.fit_resample(selected_X_train, selected_y_train)


# In[96]:


xgb_model = xgb.XGBClassifier(scale_pos_weight = 10, 
                              max_depth = 5,
                              min_child_weight = 1,
                              gamma = 0.1,
                              subsample = 0.8, 
                              colsample_bytree = 0.8
                             )
xgb_model.fit(new_x, new_y)

y_preds = xgb_model.predict(selected_X_test)
y_preds2 = xgb_model.predict(selected_X_train)

print_metrics(selected_y_test, y_preds)
print_metrics(selected_y_train, y_preds2)


# In[138]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 10, 10),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(new_x, new_y)
    y_preds = bst.predict(selected_X_test)
    f1 = f1_score(selected_y_test, y_preds)
    return f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=6000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[94]:


selected_X_train


# In[21]:


params = {
    'max_delta_step': 9, 
    'scale_pos_weight': 10, 
    'booster': 'gbtree', 
    'lambda': 4.239132807826713e-08, 
    'alpha': 5.413105889764577e-05, 
    'subsample': 0.6537500121625592, 
    'colsample_bytree': 0.6284741735703504,
    'max_depth': 9, 'min_child_weight': 10, 
    'eta': 0.1350018298900107, 
    'gamma': 0.007256337233959351, 
    'grow_policy': 'depthwise'
}

xgb_model = xgb.XGBClassifier(**params)


# In[22]:


xgb_model.fit(new_x, new_y)


# In[23]:


y_preds = xgb_model.predict(selected_X_test)
y_preds2 = xgb_model.predict(selected_X_train)


# In[24]:


print_metrics(selected_y_test, y_preds)
print()
print_metrics(selected_y_train, y_preds2)


# Accuracy: 0.8658578936401732
# Precision: 0.09851654621529099
# Recall: 0.32951653944020354
# F1-score: 0.15168374816983896
# AUC: 0.6078158608399691
# 
# Accuracy: 0.8942480378635286
# Precision: 0.2131971465629053
# Recall: 0.7342266890005583
# F1-score: 0.33044352305566027
# AUC: 0.8171858746182821

# In[41]:


params2 = {
    'max_delta_step': 9, 
    'scale_pos_weight': 10, 
    'booster': 'gbtree', 
    'lambda': 4.239132807826713e-08, 
    'alpha': 5.413105889764577e-05, 
    'subsample': 0.6537500121625592, 
    'colsample_bytree': 0.6284741735703504,
    'max_depth': 4, 
    'min_child_weight': 1, 
    'eta': 0.1350018298900107, 
    'gamma': 0.007256337233959351, 
    'grow_policy': 'depthwise'
}

xgb_model2 = xgb.XGBClassifier(**params2)
xgb_model2.fit(new_x, new_y)


# In[42]:


y_preds = xgb_model2.predict(selected_X_test)
y_preds2 = xgb_model2.predict(selected_X_train)

print_metrics(selected_y_test, y_preds)
print()
print_metrics(selected_y_train, y_preds2)


# In[48]:


over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

new_x, new_y = pipeline.fit_resample(selected_X_train, selected_y_train)


# In[44]:


xgb_model3 = xgb.XGBClassifier()

xgb_model3.fit(new_x, new_y)


# In[45]:


y_preds = xgb_model3.predict(selected_X_test)
y_preds2 = xgb_model3.predict(selected_X_train)

print_metrics(selected_y_test, y_preds)
print()
print_metrics(selected_y_train, y_preds2)


# In[ ]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 10, 10),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(new_x, new_y)
    y_preds = bst.predict(selected_X_test)
    f1 = f1_score(selected_y_test, y_preds)
    return f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, timeout=6000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[98]:


over = SMOTE(sampling_strategy=0.1)
under = TomekLinks()
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

new_x, new_y = pipeline.fit_resample(selected_X_train, selected_y_train)


# In[99]:


xgb_model4 = xgb.XGBClassifier(scale_pos_weight = 10)

xgb_model4.fit(new_x, new_y)


# In[100]:


y_preds = xgb_model4.predict(selected_X_test)
y_preds2 = xgb_model4.predict(new_x)

print_metrics(selected_y_test, y_preds)
print()
print_metrics(new_y, y_preds2)


# In[104]:


selected_X


# In[102]:


cv_ori=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring=['accuracy','precision_macro','recall_macro', 'roc_auc']
scores_ori = cross_validate(xgb_model4, selected_X, selected_y, scoring=scoring, cv=cv_ori, n_jobs=-1)


# In[103]:


print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))


# In[106]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 26, 28),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    
    bst = xgb.XGBClassifier(**param)
    
    cv_ori=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scoring=['roc_auc']
    scores_ori = cross_validate(bst, selected_X, selected_y, scoring=scoring, cv=cv_ori, n_jobs=-1)
    
    return np.mean(scores_ori['test_roc_auc'])

study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=1000, timeout=6000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[110]:


selected_y.value_counts()


# In[113]:


138822/5154


# In[114]:


best_params = {
    'max_delta_step': 7, 
    'scale_pos_weight': 27, 
    'booster': 'gbtree',
    'lambda': 0.018012543617705296,
    'alpha': 3.1587833178021335e-07, 
    'subsample': 0.58406851652986,
    'colsample_bytree': 0.5824779324896044,
    'max_depth': 5, 'min_child_weight': 10, 
    'eta': 0.07301419907048093, 
    'gamma': 3.742039792338082e-08, 
    'grow_policy': 'depthwise'
}


# In[115]:


xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(selected_X_train, selected_y_train)


# In[117]:


cv_ori=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
scoring=['accuracy','precision_macro','recall_macro', 'roc_auc']
scores_ori = cross_validate(xgb_model, selected_X, selected_y, scoring=scoring, cv=cv_ori, n_jobs=-1)


# In[118]:


print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))


# In[119]:


over = SMOTE(sampling_strategy=0.1)
under = TomekLinks()
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

sampled_X, sampled_y = pipeline.fit_resample(selected_X, selected_y)


# In[123]:


13882/137967


# In[124]:


xgb_model2 = xgb.XGBClassifier(**best_params)
xgb_model2.set_params(scale_pos_weight = 10)


# In[127]:


cv_ori=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
scoring=['accuracy','precision_macro','recall_macro', 'roc_auc']
scores_ori = cross_validate(xgb_model2, sampled_X, sampled_y, scoring=scoring, cv=cv_ori, n_jobs=-1)


# In[128]:


print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))


# In[129]:


over = SMOTE(sampling_strategy=0.2)
under = TomekLinks()
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

sampled_X2, sampled_y2 = pipeline.fit_resample(selected_X, selected_y)


# In[130]:


xgb_model3 = xgb.XGBClassifier(**best_params)
xgb_model3.set_params(scale_pos_weight = 5)


# In[131]:


cv_ori=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
scoring=['accuracy','precision_macro','recall_macro', 'roc_auc']
scores_ori = cross_validate(xgb_model3, sampled_X2, sampled_y2, scoring=scoring, cv=cv_ori, n_jobs=-1)


# In[132]:


print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))


# In[133]:


over = SMOTE(sampling_strategy=0.5)
under = TomekLinks()
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

sampled_X3, sampled_y3 = pipeline.fit_resample(selected_X, selected_y)


# In[134]:


xgb_model4 = xgb.XGBClassifier(**best_params)
xgb_model4.set_params(scale_pos_weight = 2)


# In[139]:


cv_ori=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
scoring=['accuracy','precision_macro','recall_macro', 'roc_auc', 'f1']
scores_ori = cross_validate(xgb_model4, sampled_X3, sampled_y3, scoring=scoring, cv=cv_ori, n_jobs=-1)


# In[140]:


print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))
print('Mean F1-score: %.4f' % np.mean(scores_ori['test_f1']))


# In[144]:


X = sampled_X3
y = sampled_y3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[147]:


xgb_model = xgb_model4


# In[148]:


xgb_model.fit(X_train, y_train)


# In[150]:


y_preds = xgb_model.predict(X_test)
y_preds2 = xgb_model.predict(X_train)

print_metrics(y_test, y_preds)
print_metrics(y_train, y_preds2)


# In[152]:


# filename = "xgb_model_79_auc.pickle"

# pickle.dump(xgb_model, open(filename, "wb"))

# loaded_model = pickle.load(open(filename, "rb"))


# In[156]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 2),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
    
    bst = xgb.XGBClassifier(**param)
    
    cv_ori=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    scoring=['roc_auc']
    scores_ori = cross_validate(bst, X, y, scoring=scoring, cv=cv_ori, n_jobs=-1)
    
    return np.mean(scores_ori['test_roc_auc'])

study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=500, timeout=1200)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[159]:


best_params = {'max_delta_step': 9,
 'scale_pos_weight': 2,
 'booster': 'dart',
 'lambda': 0.47831081744229986,
 'alpha': 1.1737891241000038e-08,
 'subsample': 0.5314379175798256,
 'colsample_bytree': 0.9500187766381805,
 'max_depth': 9,
 'min_child_weight': 5,
 'eta': 0.0057934334574440095,
 'gamma': 1.8449621236956727e-08,
 'grow_policy': 'depthwise',
 'sample_type': 'uniform',
 'normalize_type': 'tree',
 'rate_drop': 0.014710594833580587,
 'skip_drop': 0.0003287816633081145}


# In[180]:


xgb_model = xgb.XGBClassifier(**best_params)


# In[181]:


xgb_model.fit(X_train, y_train)


# In[182]:


y_preds = xgb_model.predict(X_test)

print_metrics(y_test, y_preds)


# In[184]:


# filename = "xgb_model_79_auc_better.pickle"

# save model
# pickle.dump(xgb_model, open(filename, "wb"))

# load model
# loaded_model = pickle.load(open(filename, "rb"))


# In[ ]:


cv_ori=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
scoring=['accuracy','precision_macro','recall_macro', 'roc_auc']
scores_ori = cross_validate(xgb_model, X, y, scoring=scoring, cv=cv_ori, n_jobs=-1)


# In[ ]:


print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))


# Mean Accuracy: 0.7940
# Mean Precision: 0.7727
# Mean Recall: 0.7958
# Mean ROC-AUC: 0.8767

# In[173]:





# In[175]:


thresholds_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

def adjust_threshold(pred_probs, threshold):
    return [1 if prob >= threshold else 0 for prob in pred_probs]


# In[176]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []
y_proba = xgb_model.predict_proba(X_test)[:, 1]

for thresh in thresholds_list:
    y_preds = adjust_threshold(y_proba, thresh)
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))


# In[177]:


data = {
    'Thresholds' : thresholds_list,
    'Accuracy' : accuracy_list,
    'Precision' : precision_list,
    'F1-scire' : f1_list,
    'ROC-AUC score' : auc_list,
    'Recall score' : recall_list
}

pd.DataFrame(data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[72]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 26, 28),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(new_x, new_y)
    y_preds = bst.predict(selected_X_test)
    roc_auc = roc_auc_score(selected_y_test, y_preds)
    return roc_auc

study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=1000, timeout=6000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[ ]:





# In[111]:


xgb_model = xgb.XGBClassifier(objective="binary:logistic", scale_pos_weight = 30.561026715557883, random_state=42)
xgb_model.fit(X_train, y_train)


# In[112]:


y_pred = xgb_model.predict(X_test)


# In[113]:


print_metrics(y_pred, y_test)


# In[120]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 27, 28),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(X_train, y_train)
    y_preds = bst.predict(X_test)
    f1 = f1_score(y_test, y_preds)
    return f1


# In[121]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, timeout=600)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[20]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 27, 28),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    roc_auc = roc_auc_score(y_test, pred_labels)
    return roc_auc


# In[21]:


study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=1000, timeout=6000)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[132]:


trial.params


# In[133]:


xgb_model = xgb.XGBClassifier(**trial.params)
xgb_model.fit(X_train, y_train)


# In[134]:


preds = xgb_model.predict(X_test)


# In[135]:


print_metrics(preds, y_test)


# In[139]:


xgb.plot_importance(xgb_model)
plt.show()


# In[147]:


thresholds = sort(xgb_model.feature_importances_)


# In[148]:


for thresh in thresholds:
 # select features using threshold
    selection = SelectFromModel(xgb_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
 # train model
    selection_model = xgb.XGBClassifier(**trial.params)
    selection_model.fit(select_X_train, y_train)
 # eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    
    print_metrics(y_test, predictions)
    print(thresh, select_X_train.shape[1])
    print()


# # FEATURE SELECTION

# Accuracy: 0.7948047137267613
# Precision: 0.08503130335799658
# Recall: 0.4751908396946565
# F1-score: 0.14425026552090373
# AUC: 0.6410335880797109
# 0.0071661402 48
# 
# Accuracy: 0.7972819669853911
# Precision: 0.08483587609801202
# Recall: 0.4669211195928753
# F1-score: 0.14358372456964005
# AUC: 0.6383403080004693
# 0.0 59

# In[ ]:





# In[149]:


# selection = SelectFromModel(xgb_model, threshold=0.0071661402, prefit=True)
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)


# In[12]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 27, 28),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(X_train, y_train)
    y_preds = bst.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
    return roc_auc


# In[13]:


study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=500, timeout=1200)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[14]:


trial.params


# In[15]:


bst = xgb.XGBClassifier(**trial.params)
bst.fit(X_train, y_train)
y_preds = bst.predict(X_test)


# In[16]:


print_metrics(y_test, y_preds)


# In[17]:


# import pickle


# In[14]:


# filename = "66_auc_only_ht_model.pickle"

# pickle.dump(bst, open(filename, "wb"))


# In[35]:


filename = "66_auc_only_ht_model.pickle"
xgb_model = pickle.load(open(filename, "rb"))


# # Oversampling the minority class

# In[36]:


oversample = SMOTE()
ovs_x, ovs_y = oversample.fit_resample(X_train, y_train)


# In[44]:


test_model = xgb.XGBClassifier()


# In[45]:


test_model.fit(ovs_x, ovs_y)


# In[46]:


y_preds = test_model.predict(X_test)


# In[47]:


print_metrics(y_test, y_preds)


# In[48]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 27, 28),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(ovs_x, ovs_y)
    y_preds = bst.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
    return roc_auc


# In[49]:


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, timeout=1200)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[50]:


bst_model = xgb.XGBClassifier(**trial.params)


# In[51]:


bst_model.fit(ovs_x, ovs_y)


# In[52]:


y_preds = bst_model.predict(X_test)

print_metrics(y_test, y_preds)


# # Over and undersampling

# In[12]:


test_model = xgb.XGBClassifier()
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('over', over), ('under', under)]
pipeline = Pipeline(steps=steps)

new_x, new_y = pipeline.fit_resample(X_train, y_train)


# In[13]:


bst_model = xgb.XGBClassifier()

bst_model.fit(new_x, new_y)


# In[14]:


y_preds = bst_model.predict(X_test)


# In[15]:


print_metrics(y_test, y_preds)


# In[80]:


new_y.value_counts()


# In[71]:


19440/9720


# In[16]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 3),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    bst = xgb.XGBClassifier(**param)
    bst.fit(new_x, new_y)
    y_preds = bst.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
    return roc_auc


# In[17]:


study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
# study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=500, timeout=1800)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[18]:


bst_model = xgb.XGBClassifier(**trial.params)


# In[19]:


bst_model.fit(new_x, new_y)


# In[20]:


y_preds = bst_model.predict(X_test)

print_metrics(y_test, y_preds)


# In[22]:


thresholds_ft = sort(bst_model.feature_importances_)


# In[30]:


def objective_selection(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 3),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    cur = xgb.XGBClassifier(**param)
    cur.fit(select_X_train, y_train)
    y_preds = cur.predict(select_X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
    return roc_auc


# In[31]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []
feature_number = []

for thresh in thresholds_ft:
    selection = SelectFromModel(bst_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_selection, n_trials=50, timeout=600)
    
    selection_model = xgb.XGBClassifier(**trial.params)
    selection_model.fit(select_X_train, y_train)
    
    y_preds = selection_model.predict(select_X_test)
    
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))
    feature_number.append(select_X_train.shape[1])
    
    print_metrics(y_test, y_preds)
    print(thresh, select_X_train.shape[1])
#     print()


# In[36]:


data = {
    'Number of features' : feature_number,
    'Accuracy' : accuracy_list,
    'Precision' : precision_list,
    'F1-scire' : f1_list,
    'ROC-AUC score' : auc_list,
    'Recall score' : recall_list
}


# In[37]:


pd.DataFrame(data)


# In[39]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []
feature_number = []

for thresh in thresholds_ft:
    selection = SelectFromModel(bst_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    
    if select_X_train.shape[1] == 48:
        print(thresh)
        break


# # The right threshold is: 0.011715805
# 

# In[45]:


plot_importance(bst_model)
plt.show()


# In[46]:


selection = SelectFromModel(bst_model, threshold=0.011715805, prefit=True)
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)


# In[48]:


xgb_model = xgb.XGBClassifier()
xgb_model.fit(select_X_train, y_train)


# In[49]:


y_preds = xgb_model.predict(select_X_test)


# In[53]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 3),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    cur = xgb.XGBClassifier(**param)
    cur.fit(select_X_train, y_train)
    y_preds = cur.predict(select_X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
    return roc_auc


# In[54]:


study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=500, timeout=1200)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


thresholds_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]


# In[48]:


def adjust_threshold(pred_probs, threshold):
    return [1 if prob >= threshold else 0 for prob in pred_probs]


# In[50]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []

for thresh in thresholds_list:
    y_proba = xgb_loaded_model.predict_proba(X_test)[:, 1]
    y_preds = adjust_threshold(y_proba, thresh)
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))


# In[61]:


data = {
    'Thresholds' : thresholds_list,
    'Accuracy' : accuracy_list,
    'Precision' : precision_list,
    'F1-scire' : f1_list,
    'ROC-AUC score' : auc_list,
    'Recall score' : recall_list
}


# In[62]:


pd.DataFrame(data)


# In[80]:


thresholds_ft = sort(xgb_loaded_model.feature_importances_)


# In[84]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []
feature_number = []

for thresh in thresholds_ft:
 # select features using threshold
    selection = SelectFromModel(xgb_loaded_model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
 # train model
    selection_model = xgb.XGBClassifier(**trial.params)
    selection_model.fit(select_X_train, y_train)
 # eval model
    select_X_test = selection.transform(X_test)
    y_preds = selection_model.predict(select_X_test)
    
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))
    feature_number.append(select_X_train.shape[1])
    
    print_metrics(y_test, y_preds)
    print(thresh, select_X_train.shape[1])
#     print()


# In[ ]:


data = {
    'Threshold' : thresholds_ft,
    'Number of feature' : feature_number,
    'Thresholds' : thresholds_list,
    'Accuracy' : accuracy_list,
    'Precision' : precision_list,
    'F1-scire' : f1_list,
    'ROC-AUC score' : auc_list,
    'Recall score' : recall_list
}

pd.DataFrame(data)


# In[51]:


y_preds_tests = bst.predict(select_X_train)
print_metrics(y_train, y_preds_tests)


# In[176]:





# In[182]:


y_proba = bst.predict_proba(select_X_test)[:, 1]


# In[193]:


print_metrics(y_test, adjust_threshold(y_proba, 0.4))


# In[ ]:





# # 66% AUC Best performance so far

# To try:
# - SMOTE, undersampling method
# - Threshold adjustment
# - Bagging, boosting

# In[164]:


xgb_model = xgb.XGBClassifier(**trial.params)
xgb_model.fit(select_X_train, y_train)


# In[165]:


preds = xgb_model.predict(select_X_test)
print_metrics(preds, y_test)


# In[195]:


get_ipython().system('pip install imbalanced-learn')


# In[199]:


import imblearn
from imblearn.under_sampling import NearMiss


# In[219]:


undersample = NearMiss(version = 3, n_neighbors_ver3 = 3, sampling_strategy = 0.1)
# transform the dataset
un_X_train, un_y_train = undersample.fit_resample(X_train, y_train)


# In[220]:


print(un_y_train.value_counts())
print(y_train.value_counts())


# In[ ]:





# In[222]:


bst = xgb.XGBClassifier(**trial.params)
bst.fit(un_X_train, un_y_train)
y_preds = bst.predict(X_test)
roc_auc = roc_auc_score(y_test, y_preds)
print(roc_auc)


# In[ ]:





# In[ ]:





# In[ ]:





# In[118]:


bst = xgb.train(trial.params, dtrain)
preds = bst.predict(dvalid)
pred_labels = np.rint(preds)
f1 = f1_score(y_test, pred_labels)

print(f1)


# In[90]:


bst = xgb.XGBClassifier(**trial.params)

bst.fit(X_train, y_train)


# In[91]:


preds = bst.predict(X_test)


# In[92]:


print_metrics(preds, y_test)


# In[96]:


trial.params


# In[94]:


bst = xgb.train(trial.params, dtrain)
preds = bst.predict(dvalid)
pred_labels = np.rint(preds)


# In[95]:


print_metrics(y_test, pred_labels)


# In[17]:


xgb_model = xgb.XGBClassifier(
    max_delta_step = 10,
    scale_pos_weight = 30,
    booster = 'dart',
#     lambda: 0.04646248587993731
    alpha = 1.9979507914709396e-05,
    subsample = 0.6760592206970389,
    colsample_bytree = 0.4903122990434893,
    max_depth = 5,
    min_child_weight = 5,
    eta = 2.0595072839157036e-08,
    gamma = 0.0270718483055683,
    grow_policy = 'depthwise',
    sample_type = 'weighted',
    normalize_type = 'forest',
    rate_drop = 1.0507111545897432e-05,
    skip_drop = 0.00017511342254744816
)

setattr(xgb_model, "lambda", 0.04646248587993731)


# In[18]:


xgb_model.fit(X_train, y_train)


# In[19]:


y_pred = xgb_model.predict(X_test)


# In[20]:


print_metrics(y_pred, y_test)


# In[45]:


xgb_model = xgb.XGBClassifier(trial.params)
xgb_model.fit(X_train, y_train)


# In[25]:


y_pred = xgb_model.predict(X_test)


# In[26]:


print_metrics(y_pred, y_test)


# In[10]:


xgb_model = xgb.XGBClassifier()

xgb_model.fit(mtrain_X, mtrain_y)


# In[11]:


y_preds = xgb_model.predict(mtest_X)

print_metrics(mtest_y, y_preds)


# In[12]:


y_train.value_counts()


# In[25]:


over = SMOTE(sampling_strategy=0.1)

new_x, new_y = over.fit_resample(X_train, y_train)


# In[26]:


new_y.value_counts()


# In[27]:


97201/19440


# In[28]:


xgb_model = xgb.XGBClassifier(scale_pos_weight = 5)

xgb_model.fit(new_x, new_y)


# In[29]:


y_preds = xgb_model.predict(X_test)

print_metrics(y_test, y_preds)
print()
y_preds2 = xgb_model.predict(X_train)
print_metrics(y_train, y_preds2)


# In[32]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 10, 10),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    cur = xgb.XGBClassifier(**param)
    cur.fit(new_x, new_y)
    y_preds = cur.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
#     y_preds2 = cur.predict(X_train)
#     roc_auc2 = roc_auc_score(y_test, y_preds2)
#     print('Test: ', roc_auc2)
    return roc_auc


# In[33]:


study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=500, timeout=1200)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[34]:


bst_model = xgb.XGBClassifier(**trial.params)

bst_model.fit(new_x, new_y)


# In[35]:


y_preds = bst_model.predict(X_test)

print_metrics(y_test, y_preds)
print()
y_preds2 = bst_model.predict(X_train)
print_metrics(y_train, y_preds2)


# In[36]:


bst_model2 = xgb.XGBClassifier(**trial.params)
bst_model2.set_params(min_child_weight = 1)

bst_model2.fit(new_x, new_y)


# In[37]:


y_preds = bst_model2.predict(X_test)

print_metrics(y_test, y_preds)
print()
y_preds2 = bst_model2.predict(X_train)
print_metrics(y_train, y_preds2)


# In[42]:


thresholds_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]


# In[43]:


def adjust_threshold(pred_probs, threshold):
    return [1 if prob >= threshold else 0 for prob in pred_probs]


# In[44]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []

for thresh in thresholds_list:
    y_proba = bst_model2.predict_proba(X_test)[:, 1]
    y_preds = adjust_threshold(y_proba, thresh)
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))


# In[45]:


data = {
    'Thresholds' : thresholds_list,
    'Accuracy' : accuracy_list,
    'Precision' : precision_list,
    'F1-scire' : f1_list,
    'ROC-AUC score' : auc_list,
    'Recall score' : recall_list
}


# In[46]:


pd.DataFrame(data)


# In[47]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []

for thresh in thresholds_list:
    y_proba = bst_model.predict_proba(X_test)[:, 1]
    y_preds = adjust_threshold(y_proba, thresh)
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))


# In[48]:


data = {
    'Thresholds' : thresholds_list,
    'Accuracy' : accuracy_list,
    'Precision' : precision_list,
    'F1-scire' : f1_list,
    'ROC-AUC score' : auc_list,
    'Recall score' : recall_list
}


# In[49]:


pd.DataFrame(data)


# In[60]:


over = SMOTE(sampling_strategy=0.2)

new_x, new_y = over.fit_resample(X_train, y_train)


# In[61]:


xgb_model = xgb.XGBClassifier(scale_pos_weight = 5)

xgb_model.fit(new_x, new_y)


# In[62]:


y_preds = xgb_model.predict(X_test)

print_metrics(y_test, y_preds)
print()
y_preds2 = xgb_model.predict(X_train)
print_metrics(y_train, y_preds2)


# In[63]:


def objective(trial):
    param = {
        "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        "verbosity": 0,
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 5, 5),
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)
        
    cur = xgb.XGBClassifier(**param)
    cur.fit(new_x, new_y)
    y_preds = cur.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_preds)
    return roc_auc

study = optuna.create_study(direction="maximize", sampler = optuna.samplers.TPESampler())
study.optimize(objective, n_trials=500, timeout=1200)

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[67]:


xgb_model3 = xgb.XGBClassifier(**trial.params)
xgb_model3.set_params(min_child_weight = 1)

xgb_model3.fit(new_x, new_y)


# In[68]:


y_preds = xgb_model3.predict(X_test)

print_metrics(y_test, y_preds)
print()
y_preds2 = xgb_model3.predict(X_train)
print_metrics(y_train, y_preds2)


# In[71]:


plot_importance(xgb_model3)
plt.show()


# In[72]:


thresholds_ft = sort(xgb_model3.feature_importances_)

accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []
feature_number = []

for thresh in thresholds_ft:
    selection = SelectFromModel(xgb_model3, threshold=thresh, prefit=True)
    select_X_train = selection.transform(new_x)
 # train model
    selection_model = xgb.XGBClassifier(**trial.params)
    selection_model.fit(select_X_train, new_y)
 # eval model
    select_X_test = selection.transform(X_test)
    y_preds = selection_model.predict(select_X_test)
    
    accuracy_list.append(accuracy_score(y_test, y_preds))
    precision_list.append(precision_score(y_test, y_preds))
    f1_list.append(f1_score(y_test, y_preds))
    auc_list.append(roc_auc_score(y_test, y_preds))
    recall_list.append(recall_score(y_test, y_preds))
    feature_number.append(select_X_train.shape[1])
    
    print_metrics(y_test, y_preds)
    print(thresh, select_X_train.shape[1])
#     print()


# In[73]:


correlation_matrix = X_train.corr()

print("Correlation Matrix:")
print(correlation_matrix)


# In[ ]:





# In[ ]:




