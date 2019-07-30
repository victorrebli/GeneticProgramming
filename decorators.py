import logging
from functools import wraps
from time import time
import pandas as pd

def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print(f'{(method.__name__, (te - ts) * 1000)}')
                  
        return result
    return timed


def exception_verified(PGException):
    def wrap(f):
        @wraps(f)
        def wrapped_f(*args, **kwargs):
           
           if not isinstance(args[1], pd.DataFrame):
               raise PGException('The "X" parameter must be a data frame')
           
           attr = args[1].select_dtypes(include=['object']).columns
           if len(attr) != 0:
              raise PGException('The following features are categoric: {attr}')
                
           return f(*args, **kwargs)            
        return wrapped_f
    return wrap                   
                       
                       
                       
                       