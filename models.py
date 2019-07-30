import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
import warnings
import logging
import collections
from datetime import datetime
warnings.filterwarnings("ignore")

class Model():
     
    def __init__(self, estimator, extra, cvs):
         self.estimator = estimator
         self.extra = extra
         self.cvs = cvs
         
    def model(self, data, y_train):
        
        module_name = self.estimator.__module__
        if 'xgboost' in module_name:
            result = self._tune_xgb(data, y_train)
        elif 'lightgbm' in module_name:
            result = self._tune_lgb(data, y_train)
            
        else:
            result = self._tune_generic(data, y_train)
            
        return result


    def _tune_xgb(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        
        for train_index, test_index in self.cvs.split(x_tr, y_tr):
            
            
            train_x, valid_x = x_tr[train_index], x_tr[test_index]
            train_y, valid_y = y_tr[train_index], y_tr[test_index]
            
            self.estimator.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric=self.extra['eval_metric'],
                    early_stopping_rounds=self.extra['early_stopping_rounds'],
                    verbose=self.extra['verbose'])
            lista_models.append(self.estimator.best_score)
            
            
            
        return np.mean(lista_models)    
            
                    
                    
                    
    def _tune_lgb(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        
        for train_index, test_index in self.cvs.split(x_tr, y_tr):
            
            
            train_x, valid_x = x_tr[train_index], x_tr[test_index]
            train_y, valid_y = y_tr[train_index], y_tr[test_index]
            
            self.estimator.fit(
                    train_x, train_y,
                    eval_set=[(valid_x, valid_y)],
                    eval_metric=self.extra['eval_metric'],
                    early_stopping_rounds=self.extra['early_stopping_rounds'],
                    verbose=self.extra['verbose'])
            lista_models.append(self.estimator.best_score_['valid_0'][self.extra['eval_metric']])
            
            
            
        return np.mean(lista_models)                    
                    
            
            
    def _tune_generic(self, x_tr, y_tr):
        
        lista_models = []
        
        x_tr = np.array(x_tr)
        
        for train_index, test_index in self.cvs.split(x_tr, y_tr):
            
            
            train_x, valid_x = x_tr[train_index], x_tr[test_index]
            train_y, valid_y = y_tr[train_index], y_tr[test_index]
            
            self.estimator.fit(
                    train_x, train_y)
            
            if self.extra['eval_metric'] == 'auc':
                
                pred = self.estimator.predict_proba(valid_x)[:, 1]
                lista_models.append(roc_auc_score(valid_y, pred))
                
        return np.mean(lista_models)      
    
    
    
    '''def _get_cvs(self, cv , y_tr):
        
        cv = check_cv(cv, y_tr, classifier=is_classifier(self.estimator))
        return cv'''
    