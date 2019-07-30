from __future__ import print_function
import operator
import numpy as np
from graphviz import Digraph
from queue import Queue
import copy

class Tree_expression:
    """
    A recursive implementation of Binary Tree
    Using links and Nodes approach.
    Modified to allow for trees to be constructed from other trees rather than always creating
    a new tree in the insertLeft or insertRight
    """
    
    def __init__(self,rootObj,label, index, parent=None):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
        self.parent = parent
        self.label = label
        self.index = index
        
    def insertLeft(self, newNode, label, itera, parent=None):

        if isinstance(newNode, Tree_expression):
            t = newNode
        else:
            t = Tree_expression(newNode, label, itera, parent)

        if self.leftChild is not None:
            t.left = self.leftChild

        self.leftChild = t

    def insertRight(self,newNode, label, itera, parent=None):
        if isinstance(newNode,Tree_expression):
            t = newNode
        else:
            t = Tree_expression(newNode, label, itera, parent)

        if self.rightChild is not None:
            t.right = self.rightChild
        self.rightChild = t

    def isLeaf(self):
        return ((not self.leftChild) and (not self.rightChild))

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild
    
    def getLabel(self):
        return self.label
    
    def getIndex(self):
        return self.index

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

    def inorder(self):
        if self.leftChild:
            self.leftChild.inorder()
        print(self.key)
        if self.rightChild:
            self.rightChild.inorder()

    def postorder(self):
        if self.leftChild:
            self.leftChild.postorder()
        if self.rightChild:
            self.rightChild.postorder()
        print(self.key)


    def preorder(self):
        print(self.key)
        if self.leftChild:
            self.leftChild.preorder()
        if self.rightChild:
            self.rightChild.preorder()                
                
    def printexp(self):
        if self.leftChild:
            if self.getRootVal() in ('cos', 'sen', 'tan', 'log'):
               print(self.key, end=' ')
               self.leftChild.printexp()
            else:
               self.leftChild.printexp()
        if self.getRootVal() not in ('cos', 'sen', 'tan', 'log'):        
            print(self.key, end=' ')
        if self.rightChild:
            self.rightChild.printexp()
            #print(')', end=' ')
    
    def postordereval(self):
        opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
        res1 = None
        res2 = None
        if self.leftChild:
            res1 = self.leftChild.postordereval()  
        if self.rightChild:
            res2 = self.rightChild.postordereval() 
        if res1 and res2:
            return opers[self.key](res1,res2) 
        else:
            return self.key
        
        
class ParseTree(object):
    """
    The goal of this class is auxiliar the create and manipulate of tree object.
    """
    def __init__(self, problem):
        self.graph = Digraph('G', filename='process1.gv',  node_attr={'shape': 'circle', 'height': '.1'})
        self.list_opera = []
        self.problem = problem
        
    def buildParseTree(self, fpexp, cols):
        """
        Create the tree from the expression.
        
        Parameters
        ----------
        fpexp: string
            Expression that will be codified into a tree object
        cols: list
            List of columns that will be codified into a tree object
        """
        fplist = fpexp.split()
        pStack = []
        itera = 0
        eTree = Tree_expression('','name' + str(itera), itera, None)
        itera +=1
        pStack.append(eTree)
        currentTree = eTree
        
        for idx,i in enumerate(fplist):
            if i == '(':
                currentTree.insertLeft('', 'name' + str(itera), itera, currentTree)
                itera +=1
                pStack.append(currentTree)
                currentTree = currentTree.getLeftChild()
            elif i in ['cos', 'sen', 'tan', 'log']:
                
                currentTree.setRootVal(i)
                
            elif i not in ['+', '-', '*', '/', ')']:
                if i in cols:
                    currentTree.setRootVal(i)
                    parent = pStack.pop()
                    currentTree = parent
                else:
                    currentTree.setRootVal(int(i))
                    parent = pStack.pop()
                    currentTree = parent
            elif i in ['+', '-', '*', '/']:
                currentTree.setRootVal(i)
                currentTree.insertRight('','name' + str(itera), itera, currentTree)
                itera +=1
                pStack.append(currentTree)
                currentTree = currentTree.getRightChild()
            elif i == ')':
                currentTree = pStack.pop()
            else:
                raise ValueError
              
        if eTree.getRootVal() != '':
            return eTree
        else:
            eTree = eTree.leftChild
            eTree.parent = None
            eTree.index = 0
            eTree.label = 'name0'
            return eTree

    def height(self, tree):
        
         """
         Calculate the deep of the tree
         """
         if tree == None:
            return -1
         else:
            return 1 + max(self.height(tree.leftChild),self.height(tree.rightChild))

    def evaluate(self, parseTree):
          """
          This functions is used to calculate the expression and return a scalar.
          """
          opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv, 
                   'cos': np.cos, 'sen': np.sin, 'tan': np.tan, 'log': np.log}
    
          leftC = parseTree.getLeftChild()
          rightC = parseTree.getRightChild()
          if leftC and rightC: 
              fn = opers[parseTree.getRootVal()]
              try:  
                  return fn(self.evaluate(leftC),self.evaluate(rightC))
              except:
                  return 0
          else:
              if parseTree.getRootVal() in ['cos', 'sen', 'tan', 'log']:
                 fn = opers[parseTree.getRootVal()]
                 try:
                     return fn(self.evaluate(leftC))
                 except:
                     return 0   
              else:
                 if parseTree.getRootVal() in self.problem.cols_names:
                     return self.problem.data[parseTree.getRootVal()]
                 else:
                     return parseTree.getRootVal()   
       
    def print_tree(self, parseTree):
        """
        Call the _print_tree
        """
        self.list_opera = []
        self._print_tree(parseTree)

    def _print_tree(self, parseTree):
        
        """
        With the GraphViz, print the expression based on the label
        """
        if parseTree.getLabel() not in self.list_opera:
                self.list_opera.append(parseTree.getLabel())
                
        if parseTree.leftChild and parseTree.rightChild:
            self.list_opera.append(parseTree.leftChild.getLabel())
            self.list_opera.append(parseTree.rightChild.getLabel())
            
            self.graph.node(parseTree.getLabel(), label=str(parseTree.getRootVal()))
            self.graph.node(parseTree.leftChild.getLabel(), label=str(parseTree.leftChild.getRootVal()))
            self.graph.node(parseTree.rightChild.getLabel(), label=str(parseTree.rightChild.getRootVal()))
            
            self.graph.edge(str(parseTree.getLabel()), str(parseTree.leftChild.getLabel()))
            self.graph.edge(str(parseTree.getLabel()), str(parseTree.rightChild.getLabel()))
            self._print_tree(parseTree.leftChild)
            self._print_tree(parseTree.rightChild)
            
        elif parseTree.leftChild:
            
            self.list_opera.append(parseTree.leftChild.getLabel())
            self.graph.node(parseTree.getLabel(), label=str(parseTree.getRootVal()))
            self.graph.node(parseTree.leftChild.getLabel(), label=str(parseTree.leftChild.getRootVal()))
            self.graph.edge(str(parseTree.getLabel()), str(parseTree.leftChild.getLabel()))
            self._print_tree(parseTree.leftChild)
            
        else:
            pass
            
    def call_print_index(self, parseTree):
        
        """
        Call the _print_tree
        """
            
        self.lista_opera_index = list()
        self.print_index_list(parseTree)
    
    def print_index_list(self, parseTree):
    
        if parseTree.getIndex() not in self.lista_opera_index:
                self.lista_opera_index.append(parseTree.getIndex())
            
        if parseTree.leftChild and parseTree.rightChild:
                    try:
                        if parseTree.leftChild.getIndex() not in self.lista_opera_index:
                            self.lista_opera_index.append(parseTree.leftChild.getIndex())
                    except:
                        pass
                    try:
                        if parseTree.rightChild.getIndex() not in self.lista_opera_index:    
                            self.lista_opera_index.append(parseTree.rightChild.getIndex())
                    except:
                        pass
                    
                    self.print_index_list(parseTree.leftChild)
                    self.print_index_list(parseTree.rightChild)
        elif parseTree.leftChild:
                    try:
                        
                        if parseTree.rightChild.getIndex() not in self.lista_opera_index:   
                            self.lista_opera_index.append(parseTree.leftChild.getIndex())
                    except:
                        pass
                    self.print_index_list(parseTree.leftChild)
            
        else:
                    pass
                    
    
    def print_tree_index(self, parseTree):
        
        """
        Call the _print_tree
        """
        
        self.list_opera = []
        self._print_tree_index(parseTree)
        
        
    def _print_tree_index(self, parseTree):
        
        """
        The goal of this function is get the index of tree.
        This function is auxiliary under the crossover time.
        """
        
        if parseTree.getLabel() not in self.list_opera:
                self.list_opera.append(parseTree.getLabel())
        
        if parseTree.leftChild and parseTree.rightChild:
        
            self.list_opera.append(parseTree.leftChild.getLabel())
            self.list_opera.append(parseTree.rightChild.getLabel())
            self.graph.node(parseTree.getLabel(), label=str(parseTree.getIndex()))
            self.graph.node(parseTree.leftChild.getLabel(), label=str(parseTree.leftChild.getIndex()))
            self.graph.node(parseTree.rightChild.getLabel(), label=str(parseTree.rightChild.getIndex()))
            
            self.graph.edge(str(parseTree.getLabel()), str(parseTree.leftChild.getLabel()))
            self.graph.edge(str(parseTree.getLabel()), str(parseTree.rightChild.getLabel()))
            self._print_tree_index(parseTree.leftChild)
            self._print_tree_index(parseTree.rightChild)
            
        elif parseTree.leftChild:
         
            self.list_opera.append(parseTree.leftChild.getLabel())
            self.graph.node(parseTree.getLabel(), label=str(parseTree.getIndex()))
            self.graph.node(parseTree.leftChild.getLabel(), label=str(parseTree.leftChild.getIndex()))
            self.graph.edge(str(parseTree.getLabel()), str(parseTree.leftChild.getLabel()))
            self._print_tree_index(parseTree.leftChild)
            
        else:
            pass        
        
        
    def print_expression(self, parseTree):
        
        """
        call de _print_expression
        """
        self.exp = ' '
        self._print_expression(parseTree)
        return self.exp
        
        
    def _print_expression(self, parseTree):
        
        """
        Print on the screen the expression of the tree
        """
        
        if parseTree.leftChild:
           if parseTree.getRootVal() in ('cos', 'sen', 'tan', 'log'): 
              self.exp = self.exp + str(parseTree.key) + ' '
           self._print_expression(parseTree.leftChild) 
                
        if parseTree.getRootVal() not in ('cos', 'sen', 'tan', 'log'): 
           self.exp = self.exp + str(parseTree.key) + ' ' 
            
        if parseTree.rightChild:
                self._print_expression(parseTree.rightChild)    