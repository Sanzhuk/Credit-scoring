#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
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
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
import time
import lightgbm as lgb
import dask
import xgboost as xgb 
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier


# In[2]:


def print_metrics(y_test, y_pred):
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'Precision: {precision_score(y_test, y_pred)}')
    print(f'Recall: {recall_score(y_test, y_pred)}')
    print(f'F1-score: {f1_score(y_test, y_pred)}')
    print(f'AUC: {roc_auc_score(y_test, y_pred)}')
    
def print_cross_val(model, XXX, yyy):
    scores_ori = cross_validate(model, XXX, yyy, scoring=scoring, cv=cv_ori, n_jobs=-1)
    print('Mean Accuracy: %.4f' % np.mean(scores_ori['test_accuracy']))
    print('Mean Precision: %.4f' % np.mean(scores_ori['test_precision_macro']))
    print('Mean Recall: %.4f' % np.mean(scores_ori['test_recall_macro']))
    print('Mean ROC-AUC: %.4f' % np.mean(scores_ori['test_roc_auc']))
    print('Mean F1-score: %.4f' % np.mean(scores_ori['test_f1']))


# In[3]:


df_train = pd.read_parquet('P03_train.pq')
df_test = pd.read_parquet('P03_test.pq')

balanced_trainX = pd.read_csv('balanced_train_X.csv').drop('Unnamed: 0', axis = 'columns')
balanced_trainy = pd.read_csv('balanced_train_y.csv').drop('Unnamed: 0', axis = 'columns')

balanced_trainX_fs = pd.read_csv('balanced_train_X_feature_selected.csv').drop('Unnamed: 0', axis = 'columns')
balanced_trainy_fs = pd.read_csv('balanced_train_y_feature_selected.csv').drop('Unnamed: 0', axis = 'columns')
testX_fs = pd.read_csv('balanced_test_X_feature_selected.csv').drop('Unnamed: 0', axis = 'columns')
testy_fs = pd.read_csv('balanced_test_y_feature_selected.csv').drop('Unnamed: 0', axis = 'columns')


# In[4]:


df_train = df_train.drop(columns = ['id'])
df_train = df_train.drop_duplicates()

X = df_train.drop(columns = ['flag'])
y = df_train['flag']


# In[5]:


original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[6]:


loaded_model = pickle.load(open('lgbm_cb_xgb_ensemble.pickle', "rb"))

X_fit = pd.read_csv('balanced_train_X.csv').drop('Unnamed: 0', axis = 1)
y_fit = pd.read_csv('balanced_train_y.csv').drop('Unnamed: 0', axis = 1)


# In[8]:


loaded_model.fit(X_fit, y_fit)


# In[10]:


y_preds = loaded_model.predict(original_Xtest)


# In[11]:


print_metrics(original_ytest, y_preds)


# In[9]:


predictions = loaded_model.predict(df_test.drop('id', axis = 1))


# In[13]:


predictions_df = pd.DataFrame({
    'id' : df_test['id'],
    'prediction' : predictions
})


# In[16]:


predictions_df.to_csv('P03_predictions.csv')


# In[ ]:





# In[ ]:





# In[25]:


LGB1 = lgb.LGBMClassifier()


# In[26]:


LGB1.fit(balanced_trainX, balanced_trainy)

LGB1.set_params(learning_rate = 0.01)
LGB1.set_params(max_depth = 7)
LGB1.set_params(num_leaves = 120)
LGB1.set_params(n_estimators = 7000)


# In[27]:


y_preds = LGB1.predict(original_Xtest)
y_preds_train = LGB1.predict(original_Xtrain)

print_metrics(original_ytest, y_preds)
print_metrics(original_ytrain, y_preds_train)


# In[ ]:





# In[9]:


LGB2 = lgb.LGBMClassifier()


# In[10]:


LGB2.fit(balanced_trainX_fs, balanced_trainy_fs)


# In[51]:


y_preds = LGB2.predict(testX_fs)

print_metrics(testy_fs, y_preds)


# In[41]:


def objective(trial):
    param_grid = {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "feature_fraction": trial.suggest_float(
            "feature_fraction", 0.2, 0.95, step=0.1
        ),
    }
    optuna_model = lgb.LGBMClassifier(**param_grid)
    optuna_model.fit(balanced_trainX, balanced_trainy)
    
    y_preds = optuna_model.predict(original_Xtest)
    f1 = f1_score(original_ytest, y_preds)
    
    return f1

lgm_study = optuna.create_study(direction = 'maximize')


# In[ ]:


lgm_study.optimize(objective, n_trials=100, timeout=1200)


# In[45]:


xgb_params = {'max_delta_step': 2, 
              'scale_pos_weight': 1, 
              'booster': 'gbtree', 
              'lambda': 4.298649239743101e-06, 
              'alpha': 5.326772199280824e-08, 
              'subsample': 0.5830491654725315, 
              'colsample_bytree': 0.46628362470420875,
              'max_depth': 5, 
              'min_child_weight': 10,
              'eta': 7.316324094541507e-07,
              'gamma': 0.0008214336727059516, 
              'grow_policy': 'depthwise'
             }

xgb_model = xgb.XGBClassifier(**xgb_params)


# In[46]:


cb_params = {'learning_rate': 0.030897543318171388,      
             'max_depth': 5, 
             'subsample': 0.6929107091543787, 
             'colsample_bylevel': 0.8580226286109289, 
             'min_child_samples': 10, 
             'reg_lambda': 0.6061297350286519,
             'n_estimators': 884
            }
cb_model = CatBoostClassifier(**cb_params)


# In[47]:


lgb_params = {'learning_rate' : 0.01,
              'max_depth' : 7,
              'num_leaves' : 120,
              'n_estimators' : 7000
            }
lgb_model = lgb.LGBMClassifier(**lgb_params)


# In[39]:


models = [
    ('LGB', lgb_model),
    ('XGB', xgb_model),
    ('CB', cb_model)
]
voter = VotingClassifier(models, voting = 'soft')


# In[42]:


prediction = voter.fit(balanced_trainX, balanced_trainy).predict_proba(original_Xtest)[:,1]


# In[44]:


prediction


# In[48]:


thresholds_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                  0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

def adjust_threshold(pred_probs, threshold):
    return [1 if prob >= threshold else 0 for prob in pred_probs]


# In[50]:


accuracy_list = []
precision_list = []
f1_list = []
auc_list = []
recall_list = []

for thresh in thresholds_list:
    y_preds = adjust_threshold(prediction, thresh)
    accuracy_list.append(accuracy_score(original_ytest, y_preds))
    precision_list.append(precision_score(original_ytest, y_preds))
    f1_list.append(f1_score(original_ytest, y_preds))
    auc_list.append(roc_auc_score(original_ytest, y_preds))
    recall_list.append(recall_score(original_ytest, y_preds))


# In[51]:


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




