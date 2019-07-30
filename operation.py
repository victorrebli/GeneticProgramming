import numpy as np
import inspect
import sys

import copy
import pandas as pd
from scipy import sparse
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier
from sklearn.metrics import roc_auc_score
import warnings
import logging
import collections
from datetime import datetime
import utils
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
    
    
class OperationException(Exception):
                                                 
      def __init__(self,message):
         super(OperationException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message) 
    

class OperationFactory():
    
    
    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
                
        else:
               raise FitnessException(f'Unsupported operation function - {type_name}')
                

                
                
class  OperationOO():
    
    def crossover_1(self, pop, cho_chosses):
            list_ch = {}
           
            for exp_ in cho_chosses:
                list_ch[exp_] = [copy.deepcopy(pop[exp_][0]), copy.deepcopy(pop[exp_][1])]

            point_cutoff = utils.search_index(list_ch)
            rootLogger.info((f'Cut point of chromosomes: {point_cutoff} \n '))
            parses_cutoff = utils.parses_trees(list_ch, point_cutoff)
            ## faz a mutacao e retorna o root
            index = np.random.choice(np.arange(0,len(list_ch)), len(list_ch), replace=False)
            cho_alter = {}
            nl = '\n'
            for num in np.arange(0, len(index), 2):
       
                rootLogger.info((f'Cross between chromosomes:{list(list_ch.keys())[index[num]]} - {list(list_ch.keys())[index[num+1]]} \n'))
        
                                 
                
                op_1 = utils.call_pars_index_mutation(copy.deepcopy(pop[list(list_ch.keys())[index[num]]][1]), 
                                                    parses_cutoff[list(list_ch.keys())[index[num + 1]]], 
                                                    point_cutoff[list(list_ch.keys())[index[num]]])

                op_1 = utils.rever_tree(op_1)
                ## refatora toda a arvore
                op_1 = utils.refactory_index(copy.deepcopy(op_1), 0)
                op_1 = utils.rever_tree(op_1,0)


                op_2 = utils.call_pars_index_mutation(copy.deepcopy(pop[list(list_ch.keys())[index[num + 1]]][1]), 
                                                    parses_cutoff[list(list_ch.keys())[index[num]]], 
                                                    point_cutoff[list(list_ch.keys())[index[num + 1]]])

                op_2 = utils.rever_tree(op_2)
                ## refatora toda a arvore
                op_2 = utils.refactory_index(copy.deepcopy(op_2), 0)
                op_2 = utils.rever_tree(op_2,0) 

                cho_alter[list(list_ch.keys())[index[num]]] = ((pop[list(list_ch.keys())[index[num]]][0],
                                                                op_1))


                cho_alter[list(list_ch.keys())[index[num+1]]] = ((pop[list(list_ch.keys())[index[num+1]]][0],
                                                                op_2))

            for key in list(cho_alter.keys()):

                pop[key] = cho_alter[key]
                
                
            return pop                     
    
                  
                  
    
    def crossover_2(self, pop, cho_chosses,best_cols):
    
            if best_cols != None: 
                list_ch = {}
                
                for exp_ in cho_chosses:
                    list_ch[exp_] = [copy.deepcopy(pop[exp_][0]), copy.deepcopy(pop[exp_][1])]

                list_best = {}
                for exp_ in best_cols:
                    list_best[exp_] = [copy.deepcopy(pop[exp_][0]), copy.deepcopy(pop[exp_][1])]


                point_cutoff_slaves = utils.search_index(list_ch)
                point_cutoff_master = utils.search_index(list_best)

                rootLogger.info((f'Cut point of chromosomes slaves: {point_cutoff_slaves} \n '))
                rootLogger.info((f'Cut point of chromosomes master: {point_cutoff_master} \n '))


                parses_cutoff_slaves = utils.parses_trees(list_ch, point_cutoff_slaves)
                parses_cutoff_master = utils.parses_trees(list_best, point_cutoff_master)

                ## faz a mutacao e retorna o root
               
                cho_alter = {}
                 
                for num in np.arange(0, len(list_ch)):
                    
                    choose_best = np.random.choice(np.arange(0, len(list_best.keys())), 1)[0]
                    
                    
                    rootLogger.info((f'Cross between chromosomes:{list(list_ch.keys())[num]} - {list(list_best.keys())[choose_best]} \n'))
                    op_1 = utils.call_pars_index_mutation(copy.deepcopy(pop[list(list_ch.keys())[num]][1]), 
                                                    parses_cutoff_master[list(list_best.keys())[choose_best]], 
                                                    point_cutoff_slaves[list(list_ch.keys())[num]])
                    op_1 = utils.rever_tree(op_1)
                    ## refatora toda a arvore
                    op_1 = utils.refactory_index(copy.deepcopy(op_1), 0)
                    op_1 = utils.rever_tree(op_1,0)

                    cho_alter[list(list_ch.keys())[num]] = ((pop[list(list_ch.keys())[num]][0],
                                                                op_1))
            else:
                
                 return self.crossover_1(pop, cho_chosses)
                   
            for key in list(cho_alter.keys()):

                pop[key] = cho_alter[key]
                
            return pop                            
                  
                  
class Operation_1(OperationOO):
    
    name = 'crossover'
    
    def __call__(self, *args, **kargs):
        return self.crossover_1(*args)
    
class Operation_2(OperationOO):
    
    name = 'crossover_top_n'
    
    def __call__(self, *args, **kargs):
        return self.crossover_2(*args)        