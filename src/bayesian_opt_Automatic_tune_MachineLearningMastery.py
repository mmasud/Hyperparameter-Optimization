# Evaluate an svm for the ionosphere dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from skopt import space
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import BayesSearchCV

"""
We will tune the following hyperparameters of the SVM model:

- C, the regularization parameter.
- kernel, the type of kernel used in the model.
- degree, used for the polynomial kernel.
- gamma, used in most other kernels.
"""
# define search space
params = dict()
params['C'] = (1e-6, 100.0, 'log-uniform')
params['gamma'] = (1e-6, 100.0, 'log-uniform')
params['degree'] = (1,5)
params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']

# define the function used to evaluate a given configuration
     
if __name__ == '__main__':

    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
    df= pd.read_csv(url)
    # split into input and output elements
    data= df.values
    X, y= data[:, :-1], data[:, -1]
    print(X.shape, y.shape)

    # define model model
    # model = SVC()
    # define cross validatin split
    cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the search
    search = BayesSearchCV(estimator=SVC(), search_spaces=params, n_jobs=-1, cv=cv, iid=False, scoring="accuracy")

    # perform the search
    search.fit(X, y)
    # report the best result
    print("Best score: %d" %(search.best_score_))
    print("Params: %s" %(search.best_params_))

