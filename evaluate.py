from decimal import Decimal
from random import sample, uniform, randint
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import collections
from importlib import reload
import problem
from problem import Problem
reload(problem)
import evaluate_3
from evaluate_3 import SolutionEvaluator
reload(evaluate_3)
import utils
import copy
import ETree
from ETree import Tree_expression, ParseTree
reload(ETree)


import decorators
from decorators import timeit, exception_verified
reload(decorators)

import featureselection
from featureselection import FSFactory
reload(featureselection)

import operation
from operation import OperationFactory
reload(operation)

import fitness
from fitness import FitnessFactory
reload(fitness)

import initialization
from initialization import InitializationFactory
reload(initialization)


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


class PGException(Exception):
                                                 
      def __init__(self,message):
         super(PGException, self).__init__()
         self.message = message
                                                 
      def __str__(self):
         
         return repr(self.message)                                        

                                         
class GPFactory():
                     
                   
     @staticmethod
     def build(estimator,num_particles=30, max_iter=100,initialization='default',
               fitness_method='roulette_whell', operation = 'crossover', method_fs = 'FS', 
               max_local_improvement=50,maximize_objective=True, cv = 3, fill_nan = None, 
               perc_mutation = 0.5):
                     
         """
         Factory class that receives the options and redirect to sub_classes factorys
         
         Parameters
         ----------
         estimator: object
             Estimator that will be used.
         
         num_particles: int
             Number the particles - chromosomes.
         
         max_iter: int
             The max number of iteration
         
         initialization: string
             How the particles will be initialized
             options - |default|
         
         fitness_method: string
             How the particles will be sorted based on fitness.
             option - |roulette_whell| - |roulette_linear| - |roulette_top_n|
             
         operation: string
             How the select particles based on the fitness_method will make alteration
             in you structure.
             option - |crossover| - |crossover_top_n|
             
         method_fs: string
             In the final of each iteration, a feature selection will be done
             option - |FS| - |BS|
             
         max_local_improvement: int
             The maximum number iteration without improvemment in the best particle before
             exit the train.
         
         maximize_objective: bool
             The metric will be maximize ou minimize
             Ex: AUC(maximize) - error(Minimize)
             
         cv: int
             The number of fold in the train
             
         fill_nan: int/float
             How fill the NaN that eventualy occur.
         
         perc_mutation: float - 0.0 - 1.0
             The percentage of particles that will suffer crossover
         """
         initial = InitializationFactory.build(initialization)         
         fitness = FitnessFactory.build(fitness_method)
         operat = OperationFactory.build(operation)            
         fs = FSFactory.build(method_fs)            
         return  GeneticProgramming(estimator, num_particles, max_iter, initial, 
                                    fitness, operat, fs, max_local_improvement,
                                    maximize_objective, cv, fill_nan, perc_mutation)
                     
                     
                     
class GeneticProgramming(object):
      
      def __init__(self, estimator, num_particles, max_iter,initialization,
                   fitness_method, operation, method_fs, max_local_improvement,
                   maximize_objective, cv, fill_nan, perc_mutation):
                                                 
           self.num_particles = num_particles
           self.max_iter = max_iter
           self.cv = cv
           self.evaluator_ = None
           self.estimator = estimator
           self.max_local_improvement = max_local_improvement
           self.local_improvement = 0
           self.iteration_ = 0
           self.pop_ = None
           self.maximize_objective = maximize_objective
           self.init_method_ = initialization
           self.fitness_method_ = fitness_method
           self.mutation_method_ =  operation
           self.fs_method_ = method_fs
           self.best_trees = list()
           self.fill_nan = fill_nan
           self.perc_mutation = perc_mutation
           self._verified_params()   
                  
      #@timeit
      @exception_verified(PGException)
      def fit(self, X, unused_y, **kargs):
                 
           prob = Problem(X, unused_y, self.estimator,
                          self.cv, **kargs)    
        
           self._initialize(X.columns, prob)
                                                 
           self.evaluator_ = SolutionEvaluator(prob, self.num_particles, self.fs_method_, self.maximize_objective)
           
           while not self._is_stop_criteria_accepted():
               
               self.init_search()
               
               if self.iteration_ % 10 == 0:
                  rootLogger.info((
                           f'Until Now, the best expressions are of cromossomos: \n '
                              ))
                  for idx, crho in enumerate(self.best_trees):
                        
                      chromosome_parsetree, chromosome_tree = crho[0], crho[1]
                      rootLogger.info((
                           f'Expression {idx} - {chromosome_parsetree.print_expression(chromosome_tree)}  \n '
                              ))
                        
      def _verified_params(self):
          """
          Make the firsts verifieds errors params.
          """
          if self.num_particles < 0:
               raise PGException('The "num_particles" can not be negative')
            
          if (self.perc_mutation < 0.0) | (self.perc_mutation > 1.0):
               raise PGException('The "perc_mutation" should be between 0 and 1.0')  
                
          if self.fill_nan == None:
               rootLogger.info((
                           f' You dont specified a value to fill eventualy NaN. If you estimador dont support NaN, it will be occur \
                           an error.\n '
                              ))     
            
      def _initialize(self, columns, prob):
          """
          Initialize the population
          """ 
          self.iteration_ = 0
          self.pop_ = self.init_method_.create_population(columns, self.num_particles, prob)
           
      def _is_stop_criteria_accepted(self):
          
          """
          return whether the criteria of stop was reachable.
          The maximium number of iteration or maximum number
          of iteration without improvement
          """
          max_iter_reached = self.iteration_ == self.max_iter
          return max_iter_reached
      
        
      def init_search(self):
          """
          Initializer the search of GP
          """
          rootLogger.info((f'Iteration: {self.iteration_}/{self.max_iter} \n'))
          self.fitness, best_cols, pred = self.evaluator_.evaluate(self.pop_, self.iteration_,self.fill_nan)
          #breakpoint()           
          self.fitness = collections.OrderedDict(sorted(self.fitness.items(), key=lambda x: x[1], reverse=self.maximize_objective))
          rootLogger.info((
                           f'Fitness of cromossomos sorted: {self.fitness} \n '
                              ))
          if best_cols !=  None:
              if (len(best_cols) != len(list(self.fitness.keys()))):
                   for crho in best_cols:
                     chromosome_parsetree, chromosome_tree = self.pop_[crho][0], self.pop_[crho][1]
                     rootLogger.info((
                               f'Expression of cromossomos: {crho} - {chromosome_parsetree.print_expression(chromosome_tree)}  \n '
                                  ))
                     self.best_trees.append((chromosome_parsetree,chromosome_tree))
                     if self.mutation_method_.name == 'crossover_top_n':
                           try:
                              self.fitness.pop(crho)
                           except KeyError:
                              rootLogger.info((
                               f'Key Not found: {crho} \n '
                                  ))

          num_ = int(self.perc_mutation * self.num_particles)
          if num_ % 2 != 0:
             num_ -= 1
              
          cho_chosses = self.fitness_method_(self.fitness,num_)     
          if self.mutation_method_.name == 'crossover_top_n':
              self.pop_ = self.mutation_method_(self.pop_,cho_chosses, best_cols)
          else:
              self.pop_ = self.mutation_method_(self.pop_,cho_chosses)
                
          self.iteration_ += 1
                     
    
    