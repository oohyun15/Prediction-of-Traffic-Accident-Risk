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
import time
import numpy as np
import winprocess as w
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

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import balanced_accuracy_score
from concurrent.futures import as_completed, ProcessPoolExecutor

from sklearn.metrics import accuracy_score

def make_model(param_grid):
    
    print(type(model))
    y_pred = model.fit(x_train,y_train).predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
def parallel_processing(param_grid):
    executor = ProcessPoolExecutor(max_workers = 4)
    max_score, score = 0, 0
    model = executor.submit(KNeighborsClassifier,param_grid)
    model = model.result()
    y_pred = model.fit(x_train,y_train).predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    
    

if __name__ == '__main__':
    param_range = [3,5,7]
    param_pval = [1,2]
    param_grid = [
        {'n_neighbors': param_range, 'p': param_pval}
    ]
    parallel_processing(param_grid)