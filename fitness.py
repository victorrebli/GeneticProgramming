import numpy as np
import inspect
import sys

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
    
    
class FitnessException(Exception):
                                                 
      def __init__(self,message):
         super(FitnessException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message) 
    

class FitnessFactory():
    
    
    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
                
        else:
               raise FitnessException(f'Unsupported fitness function - {type_name}')
                
                
class RouletteWhell():
    
    name = 'roulette_whell'
    
    @staticmethod
    def roulette_whell(pop, size):
    
        dat_ = pd.DataFrame(data={'keys': list(pop.keys()), 'prob': list(pop.values()) / np.sum(list(pop.values()))})
        rootLogger.info(( f' Chromosomes probabilities: {list(dat_["prob"].values)} \n '))
        cho_chosses = np.random.choice(list(dat_['keys'].values), replace=False, size=size, p=list(dat_['prob'].values))
        rootLogger.info(( f' Chromosomes that will mutate: {cho_chosses} \n '))
        return cho_chosses
                         
    def __call__(self, *args, **kargs):
        return self.roulette_whell(*args)
    
    
class RouletteLinear():
    
    name = 'roulette_linear'
    
    @staticmethod
    def roulette_linear(pop, size):
    
        pop_ = {}

        for num, name in zip(np.arange(len(pop), 0, -1), list(pop.keys())):            
            pop_[name] = num

        dat_ = pd.DataFrame(data={'keys': list(pop_.keys()), 'prob': list(pop_.values()) / np.sum(list(pop_.values()))})
        rootLogger.info(( f' Chromosomes probabilities: {list(dat_["prob"].values)} \n '))
        cho_chosses = np.random.choice(list(dat_['keys'].values), replace=False, size=size, p=list(dat_['prob'].values))
        rootLogger.info(( f' Chromosomes that will mutate: {cho_chosses} \n '))
        return cho_chosses    

    def __call__(self, *args, **kargs):
        return self.roulette_linear(*args)
        


class Roulette_top_n():
    
    name = 'roulette_top_n'
    
    @staticmethod
    def roulette_top_n(pop, size):
    
        pop_ = {}
        if len(pop) + 1 <= size:
            raise PGException(f'The len of population must be greater than {size} when routelle \
                               top n is utilized.')   
        COLS_ADD = []
        for name in list(pop.keys())[1:size]:
            COLS_ADD.append(name)                   

        rootLogger.info(( f' Chromosomes that will mutate: {COLS_ADD} \n '))
        return COLS_ADD                         


    def __call__(self, *args, **kargs):
        return self.roulette_top_n(*args)
                         
                         
