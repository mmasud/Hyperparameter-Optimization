import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import pipeline

from functools import partial
from skopt import space
from skopt import gp_minimize
from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
import optuna


def optimize(trials, x, y):
    criterion = trials.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators= trials.suggest_int("n_estimators", 100, 600)
    max_depth= trials.suggest_int("max_dept", 3, 15)
    max_features= trials.suggest_uniform("max_features", 0.01, 1.0)
    
    ##### Things will update here. such as pipeline
    
    model = ensemble.RandomForestClassifier(
        n_estimators= n_estimators,
        max_depth= max_depth,
        max_features= max_features,
        criterion= criterion,
    )
     
    kf = model_selection.StratifiedKFold(n_splits=5)

    accuracy = []
    for idx in kf.split(X=x, y= y):
        train_idx, test_idx= idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]

        xtest= x[test_idx]
        ytest = y[test_idx]


        model.fit(xtrain, ytrain)
        preds= model.predict(xtest)
        fold_acc= metrics.accuracy_score(ytest, preds)
        accuracy.append(fold_acc)
    return -1.0 * np.mean(accuracy)


if __name__ == '__main__':
    
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop('price_range', axis=1).to_numpy()
    y= df.price_range.to_numpy()

    optimization_function = partial(optimize, x=X, y=y)
    study = optuna.create_study(direction= "minimize")
    study.optimize(optimization_function, n_trials=15)