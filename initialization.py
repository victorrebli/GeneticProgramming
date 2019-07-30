from importlib import reload
import numpy as np
import inspect
import sys
import utils
reload(utils)
import pandas as pd
import copy
from scipy import sparse
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
import warnings
import logging
import collections
from datetime import datetime

import ETree
from ETree import Tree_expression, ParseTree
reload(ETree)
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
    
    
LIST_OPE = ['sum','sub','mult','div','log', 'tan', 'sen', 'cos']

class InitializationException(Exception):
                                                 
      def __init__(self,message):
         super(InitializationException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message) 
    

class InitializationFactory():
    
    
    @staticmethod
    def build(initialization):
        #breakpoint()
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and initialization == obj.name:
                    return obj()
                
        else:
               raise InitializationException(f'Unsupported initialization function - {initialization}')
                
                
class CreatePopulation():
    
    name = 'default'
    
    @staticmethod
    def create_population(cols, num_particles, prob, qtd_sub = 10):
        """
        This method create the initial population.
        
        Parameters
        ----------
        cols: list of features (columns)
            List of columns of the dataset
        
        num_particles: int
            Number of particles (chromosomes) that will be initialize
        
        prob: Object
            Object of class Problem
            
        qtd_sub: int, optional
            Number os sub_level inside the initial expression.
            For a while, this parameter is static.
            
        Returns
        -------
        list_dc: dict
            dict of list that contains for each chromosome a Tree expression and assistant
            class to manipulate the Tree.
        """
        
        opera_primary = utils.choose_operations(LIST_OPE, qtd = num_particles, qtd_sub = qtd_sub)
        list_dc = {}
        for idx, ope in enumerate(opera_primary):
            lista, dc1, dc2 = list(), list(), list()
            lista.append(ope[0])
            [lista.append(var) for var in ope[1]]
            dc = utils.create_exp(lista, cols)
            
            [dc1.append(var) if list(var.keys())[0] == 1 else dc2.append(var) for var in dc]
            dc = dc1 + dc2
            
            dc = utils.buildParseTree_tree(dc, cols)
            rootLogger.info((
                           f'Initial Cromossomos: {idx} - {dc} \n '))
            pt_1 = ParseTree(prob)
            parseTree_1 = pt_1.buildParseTree(dc, cols)
            list_dc[f'Cromossomo_{idx}'] = ((copy.deepcopy(pt_1),copy.deepcopy(parseTree_1)))
        return list_dc         

    def __call__(self, *args, **kargs):
        return self.create_population_default(*args)
                
                