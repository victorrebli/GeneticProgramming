# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:12:08 2019

@author: reblivi
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
from importlib import reload
import warnings
import logging
import collections
from datetime import datetime
warnings.filterwarnings("ignore")

import models
from models import Model
reload(models)

logFormatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')

rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

if not rootLogger.handlers:
    fileHandler = logging.FileHandler(datetime.now().strftime('PG_%d-%m_%H:%M.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    
 
def minimize_comparator(individual_solution, global_solution) -> bool:
    return individual_solution < global_solution

def maximize_comparator(individual_solution, global_solution) -> bool:
    return individual_solution > global_solution

class PException(Exception):
                                                 
      def __init__(self,message):
         super(PException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message)   
    
    
class SolutionEvaluator():
    
    
    def __init__(self, p, size_pop, method_fs, maximize_objective):
         
        self.p = p
        #self.cv = p.cv
        self.cvs = None
        self.cv = self.p.cv
        self.estimator = self.p.estimator
        self.extra = self.p.extra
        self.fitness = {}
        self.score_ofeatures = 0
        self.method_fs = method_fs
        self.models = 0
        self._setup_initialization()
        self._setup_solution_comparator(maximize_objective)
        
    
    def _setup_initialization(self):
        
           """
           Initialization the cross_validation, the model to run the estimator
           and run the model the first time to get the baseline, the score with all 
           original features.
           """
           self.cvs = self._get_cvs(self.p.label)
           self.models = Model(self.estimator, self.extra, self.cvs) 
           
           self.score_ofeatures = self.models.model(self.p.data.copy(), self.p.label.copy()) 
           rootLogger.info(( f' Score with original features: {self.score_ofeatures} \n ')) 
           
            
    def _setup_solution_comparator(self, maximize_objective):
           """
           Initialize the functions that will store if the metric should
           be maximize ou minimize
           """
           self.maximize_objective = maximize_objective
           if self.maximize_objective:
               self.is_solution_better = maximize_comparator
           else:
               self.is_solution_better = minimize_comparator
                
    def evaluate(self, pop, iteration, fill_nan):
        
        x_train = self.p.data.copy()
        y_train = self.p.label.copy()
        
        fitness = {}
        features_chr = {}
        for idx,name in enumerate(list(pop.keys())): 
            chromosome_parsetree, chromosome_tree = pop[name][0], pop[name][1]
            features = chromosome_parsetree.evaluate(chromosome_tree)
            x_train['gp_feature'] = features
            
            if x_train['gp_feature'].isnull().sum() > 0:
                rootLogger.info((f'{name}_{iteration} has {x_train.gp_feature.isnull().sum() / x_train.shape[0]} Null Lines  \n '))
                if fill_nan != None:
                    rootLogger.info(( f'Filling the NaN with {fill_nan} \n '))
                    x_train['gp_feature'].fillna(fill_nan, inplace=True)
                else:
                    rootLogger.info(( f'You dont specified a fill_nan...nothing will be done \n '))
                   
            features_chr[f'{name}_{iteration}'] = x_train['gp_feature'].values
            pred = self.models.model(x_train, y_train)
            fitness[name] = pred
        
        x_train = x_train.drop('gp_feature', axis=1)
        rootLogger.info((f'Original Features that will be use in FS Selection: {self.p.data.columns} \n '))
        best_cols, pred = self.method_fs.feature_selection(self.models, x_train, y_train,
                                                            fitness, features_chr, iteration,
                                                            self.maximize_objective, self.score_ofeatures,
                                                            self.is_solution_better)
        
        #breakpoint()                                                                                                
        if (best_cols == None) & (pred == None):
            rootLogger.info((f'This ITERATION: {iteration} - dont found any better features combinations \n '))
            rootLogger.info((f'The best metric keep doing: {self.score_ofeatures} \n '))
        else:
            self.score_ofeatures = pred
            rootLogger.info((f'BEST combinations features : {best_cols}  - METRIC - {pred} \n '))
            feat_int = pd.concat(list(map(lambda name_: pd.DataFrame(features_chr[name_], columns=[name_]), best_cols)), 
                                     axis=1)
            self.p.data = pd.concat([self.p.data,feat_int], axis=1)
            
            best_cols = [var.split('_')[0] + '_' + var.split('_')[1] for var in best_cols]
        #breakpoint()    
        return fitness, best_cols, pred
                      
        
    def _get_cvs(self, y_tr):
        
        cv = check_cv(self.cv, y_tr, classifier=is_classifier(self.estimator))
        return cv
    
                
        
        
        