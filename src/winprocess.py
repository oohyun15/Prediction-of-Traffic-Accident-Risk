import inspect
import types
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
# dummy data for debug
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
from sklearn.model_selection import train_test_split

#3
# preprocessing = pd.read_csv('../input/finalres.csv')

# preprocessing = preprocessing.iloc[np.random.permutation(len(preprocessing))]

# y = preprocessing.loc[:, preprocessing.columns == '사고내용']
# y = y.astype('float')
# x = preprocessing.loc[:, preprocessing.columns != '사고내용']

# x = x[:30000]
# y = y[:30000]
newx = pd.read_csv('../input/newx.csv')
newy = pd.read_csv('../input/newy.csv')
newy = newy.astype('float')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(newx, newy, test_size=0.2)

def execute_source(callback_imports, callback_name, callback_source, args):
    for callback_import in callback_imports:
        exec(callback_import, globals())
    exec('import time' + "\n" + callback_source)
    callback = locals()[callback_name]
    return callback(*args)


def submit(executor, callback, *args):
    callback_source = inspect.getsource(callback)
    callback_imports = list(imports(callback.__globals__))
    callback_name = callback.__name__
    future = executor.submit(
        execute_source,
        callback_imports, callback_name, callback_source, args
    )
    return future


def imports(callback_globals):
    for name, val in list(callback_globals.items()):
        if isinstance(val, types.ModuleType) and val.__name__ != 'builtins' and val.__name__ != __name__:
            import_line = 'import ' + val.__name__
            if val.__name__ != name:
                import_line += ' as ' + name
            yield import_line