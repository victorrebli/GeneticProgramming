
from __future__ import print_function
import operator
import numpy as np
from graphviz import Digraph
from queue import Queue
import copy


def call_pars_index(parseTree,search_index):
    return search_index_parsetree(parseTree,search_index)

def search_index_parsetree(parseTree, search_index):
    q = Queue()  
    q.put(parseTree)  
    while (q.empty() == False): 

        node = q.queue[0]  
        if (node.index == search_index):
            node.parent = None
            return node

        q.get() 
        if (node.leftChild != None): 
            q.put(node.leftChild)  
        if (node.rightChild != None): 
            q.put(node.rightChild)

    return False 

def call_pars_index_mutation(parseTree_O, parseTree_M, search_index):

    q = Queue()  
    q.put(parseTree_O)
    pStack = list()

    indexes = list()
    while (q.empty() == False): 

        node = q.queue[0]
        if (node.index == search_index):
            node = parseTree_M
            pStack.append(node)
            return pStack
        q.get()
        if (node.leftChild != None):
            if node.leftChild.index == search_index:
                node.leftChild = parseTree_M
                pStack.append(node)
                return pStack

            q.put(node.leftChild)  
        if (node.rightChild != None): 
            if node.rightChild.index == search_index:
                node.rightChild = parseTree_M
                pStack.append(node)
                return pStack
            q.put(node.rightChild)
    return False

def rever_tree(dc, flag = 1):
    dc_t = True
    interm = dc[0].parent
    if flag == 1:
        if interm and (dc[0].leftChild or dc[0].rightChild):
            while dc_t:
                if interm.parent:
                   interm = interm.parent
                else:
                   dc_t = False
            return interm    
        else:
            return dc[0]
    
    else:
        if interm:
            while dc_t:
                if interm.parent:
                   interm = interm.parent
                else:
                   dc_t = False
            return interm    
        else:
            return dc[0]
        
def refactory_index(parseTree, iteri):
    q = Queue()  
    q.put(parseTree)
    pstack = []
    parent = None
    while (q.empty() == False): 

        node = q.queue[0] 
        node.index = iteri
        node.label = 'name' + str(iteri)
        iteri += 1
        q.get() 
        if (node.leftChild != None):
            node.leftChild.parent = node
            q.put(node.leftChild)  
        if (node.rightChild != None): 
            q.put(node.rightChild)
            node.rightChild.parent = node
    pstack.append(node)
    return pstack 

def search_index(particles):
    roll_part = {}
    for key in particles.keys():
        pt, parsetree = particles[key] 
        pt.call_print_index(parsetree)
        roll_part[key] = np.random.choice(pt.lista_opera_index, 1)[0]
    return roll_part

def parses_trees(list_particles, lista):
    parses = {}
    for key in list_particles.keys(): 
       _, parsetree = list_particles[key]
       parse = call_pars_index(parsetree,lista[key])
       parses[key] = parse
    return parses    


def choose_operations(lista, qtd = None, qtd_sub = 5, level = 0):
    """
    Given a operations list, choose randomly for each chromosome
    what will be the composition fo expression
    
    Parameters
    ---------
    lista: List
        List of operators allows
    
    qtd: Number of chromosome.
        If 0, a number 10 will be a choose static.
        
    qtd_sub: int
        the deep of expression
    
    Returns
    -------
    cho|prof: string, list,
        Actualy, only the prof works.
        return |qtd| list of operations.
    
    """
    prof = list()
    if level == 0: 
        if qtd == None:
            qtd = np.random.choice(np.arange(0,10),1) 
        
        cho = np.random.choice(lista, qtd)
     
        for ind in list(cho):
             n = np.random.choice(np.arange(0,qtd_sub),1)  
             prof.append((ind,np.random.choice(lista, n)))
    else:
        
        cho = np.random.choice(lista, 1)
        return cho
    return prof

def create_exp(exp, cols):
    """
    Paramaters
    ----------
    
    exp: list 
        List os Operators
    
    cols: list
        List of Columns
        
    How this works?
        The goal is build a list of pre-expression that 
        more later will be afflutined     
    """
    opers = {'sum':'+','sub':'-', 'mult':'*', 'div':'/', 
              'log': 'log', 'sen': 'sen', 'cos': 'cos',
             'tan': 'tan'}
    
    lista_app = list()
    for idx, expr in enumerate(exp):
        if expr in ['sum', 'sub', 'mult', 'div']:
            lista_app.append({1: '( ' + choose_features(cols) + ' ' + opers[expr] + ' ' + choose_features(cols) + ' )'})
            
        else:
            lista_app.append({2:opers[expr]})
    return lista_app


def choose_features(cols):
    """
    Choose randomly one scalar between [0,20] or
    a col choosen randomly from cols
    """
    lista1 = np.arange(0,20)
    num = np.random.choice(lista1,1)
    qtd = np.random.choice(cols,1)
    return np.random.choice([num[0],qtd[0]],1)[0]
    
def buildParseTree_tree(lista, cols):
    """
    Build the final expression.
    
    Parameters
    ----------
    lista: list
        List of pre-expression
            key 1 means that elemantry operation(sum,minus,div...)
            key 2 means other operation(log,sen,cos...)\
    
    cols: list
        List of columns that will be randomly choosen to build 
        the final expression from the pre-expression
    
    Returns
    -------
    expre: string
        Final expression
    """
    tam = len(lista)
    flag = 0
    expre = '('
    for idx in np.arange(0, tam - 1):
        
        if (list(lista[idx].keys())[0] == 1):
            if (list(lista[idx+1].keys())[0] == 1):
                expre = expre + ' ' + '(' + ' ' + lista[idx][1] + ' ' + '+'
                flag += 1
            elif (list(lista[idx+1].keys())[0] != 1):
                expre = expre + ' ' + lista[idx][1] + ' '
            
                for i in np.arange(0,flag):
                    flag -= 1
                    expre = expre + ')' + ' '
                expre = expre + '+'   \
                
        elif (list(lista[idx].keys())[0] == 2):
                 
            if (list(lista[idx+1].keys())[0] == 2):
                  flag += 1
                  expre = expre + ' ' + lista[idx][2] + ' ' + '(' 
                
            elif (list(lista[idx+1].keys())[0] != 2):
                  expre = expre + ' ' + lista[idx][2] + ' ' + '(' + ' ' + '(' + ' ' + choose_features(cols) + ' ' + '-' + ' ' + choose_features(cols) + ' ' \
                  + ')' + ' ' + ')' + ' '
                  for i in np.arange(0,flag):
                     flag -= 1
                     expre = expre + ')' + ' ' 
                  expre = expre + '+'
                        
    if (list(lista[tam-1].keys())[0] == 1):
        if (list(lista[tam-2].keys())[0] == 2):
            expre = expre + ' ' + lista[tam-1][1] + ' '
            for i in np.arange(0,flag):
                expre = expre + ')' + ' '
        else:
            expre = expre + ' ' + lista[tam-1][1] + ' '
            for i in np.arange(0,flag):
                expre = expre + ')' + ' '
    
    elif (list(lista[tam-1].keys())[0] == 2):
        if (list(lista[tam-2].keys())[0] == 2):
           expre = expre + ' ' + lista[tam-1][2] + ' ' + '(' + ' ' + '(' + ' ' + choose_features(cols) + ' ' + '-' + ' ' + choose_features(cols) + ' ' + ')' + ' ' + ')' + ' '
           for i in np.arange(0,flag):
                expre = expre + ')' + ' ' 
        elif (list(lista[tam-2].keys())[0] != 2):
            
            expre = expre + ' ' + lista[tam-1][2] + ' ' + '(' + ' ' + '(' + ' ' + choose_features(cols) + ' ' + '-' + ' ' + choose_features(cols) + ' ' + ')' + ' ' + ')' + ' '
            for i in np.arange(0,flag):
                expre = expre + ')' + ' ' 
    expre = expre + ')'
    return expre