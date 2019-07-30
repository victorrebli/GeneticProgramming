import numpy as np
import inspect
import sys

from importlib import reload
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

import models
from models import Model
reload(models)

class FSException(Exception):
                                                 
      def __init__(self,message):
         super(FSException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message) 
    

class FSFactory():
    
    
    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
                
        else:
               raise FSException(f'Unsupported operation function - {type_name}')
                                
class FSF():
    
    name = 'FS'
    
    @staticmethod
    def feature_selection(model, x_train, y_train, fitness, features_chr, iteration, maximize_objective, best_score,
                         is_solution_better):
       
        COLS_GP = list(features_chr.keys())
        COLS_INTERM = []
        
        fitness = collections.OrderedDict(sorted(fitness.items(), key=lambda x: x[1], reverse=maximize_objective))
        flag = 0
        if not is_solution_better(fitness[list(fitness.keys())[0]],best_score):    
           return None, None
        else:
           best_score_ = fitness[list(fitness.keys())[0]]
           COLS_INTERM.append(list(fitness.keys())[0] + '_' + str(iteration))  
           COLS_GP.remove(list(fitness.keys())[0] + '_' + str(iteration))         
           while flag != 1:        
             dici_aux = {}
             for name in COLS_GP:      
                feat_int = pd.concat(list(map(lambda name_: pd.DataFrame(features_chr[name_], columns=[name_]), COLS_INTERM + [name])), 
                                     axis=1)  
                dici_aux[name] = model.model(pd.concat([x_train,feat_int],axis=1), y_train)   
             dici_aux = collections.OrderedDict(sorted(dici_aux.items(), key=lambda x: x[1], reverse=maximize_objective))
             if not is_solution_better(dici_aux[list(dici_aux.keys())[0]],best_score_):
                 flag = 1       
             else:
                 try:
                     best_score_ = dici_aux[list(dici_aux.keys())[0]]
                     COLS_INTERM.append(list(dici_aux.keys())[0])  
                     COLS_GP.remove(list(dici_aux.keys())[0])
                     if len(COLS_GP) == 0:
                            flag = 1
                 except:
                     pass 
           return COLS_INTERM, best_score_      
        
    def __call__(self, *args, **kargs):
        return self.feature_selection(*args)
    
    
class BSF():
    
    name = 'BS'
    
    @staticmethod
    def feature_selection(model, x_train, y_train, fitness, features_chr, iteration, maximize_objective, best_score,
                         is_solution_better):
       
        COLS_GP = list(features_chr.keys())
        COLS_INTERM = []
        
        fitness = collections.OrderedDict(sorted(fitness.items(), key=lambda x: x[1], reverse=maximize_objective))
        flag = 0
        
        feat_int_ = pd.concat(list(map(lambda name_: pd.DataFrame(features_chr[name_], columns=[name_]), COLS_GP)), 
                            axis=1) 
        best_score_ = model.model(pd.concat([x_train,feat_int_],axis=1), y_train) 
        
        if not is_solution_better(best_score_,best_score):    
           return None, None
        else:       
           while flag != 1:        
             dici_aux = {}
             for name in COLS_GP:      
                feat_int_ = pd.concat(list(map(lambda name_: pd.DataFrame(features_chr[name_], columns=[name_]), 
                                     list(set(COLS_GP).difference(set([name]))))), axis=1)
                
                dici_aux[name] = model.model(pd.concat([x_train,feat_int_],axis=1), y_train)   
                
             dici_aux = collections.OrderedDict(sorted(dici_aux.items(), key=lambda x: x[1], reverse=maximize_objective))
             if not is_solution_better(dici_aux[list(dici_aux.keys())[0]],best_score_):
                 flag = 1       
             else:
                 try:
                     best_score = dici_aux[list(dici_aux.keys())[0]] 
                     COLS_GP.remove(list(dici_aux.keys())[0])
                     if len(COLS_GP) == 1:
                            flag = 1
                 except:
                     pass 
           return COLS_GP, best_score_      
        
    def __call__(self, *args, **kargs):
        return self.feature_selection(*args)    