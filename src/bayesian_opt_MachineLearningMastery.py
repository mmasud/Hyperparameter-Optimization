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

"""
We will tune the following hyperparameters of the SVM model:

- C, the regularization parameter.
- kernel, the type of kernel used in the model.
- degree, used for the polynomial kernel.
- gamma, used in most other kernels.
"""
search_space= list()
search_space.append(space.Real(1e-6, 100.0, prior="log-uniform", name="C"))
search_space.append(space.Categorical(["linear", "poly", "rbf", "sigmoid"], name="kernel"))
search_space.append(space.Integer(1, 5, name="degree"))
search_space.append(space.Real(1e-6, 100.0, prior="log-uniform", name= "gamma"))

# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
    # configure the model with specific hyperparameters
    model= SVC()
    model.set_params(**params)
    # define test harness
    cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # calculate 10-fold cross validation
    result = cross_val_score(model, X, y, cv=cv, n_jobs=-1, scoring='accuracy')
    # calculate the mean of the scores
    estimate= np.mean(result)
    # convert from a maximizing score to a minimizing score
    return 1.0 - estimate # perfect skill will be (1 – accuracy)

     
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
    # # define cross validatin split
    # cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # # evaluate model
    # m_score= cross_val_score(model, X, y=y, scoring="accuracy", cv=cv, n_jobs= -1, error_score="raise")
    # print("Accuracy: %.3f (%.3f)" % (np.mean(m_score), np.std(m_score)))

    
    # perform optimization
    result = gp_minimize(evaluate_model, search_space,
                        verbose=10)
    # Summarizing findings
    print("Best Accuracy: %.3f" % (1.0- result.fun))
    print("Best Parameters: %s" % (result.x))