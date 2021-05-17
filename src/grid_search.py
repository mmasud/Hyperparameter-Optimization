import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import pipeline




if __name__ == '__main__':
    df = pd.read_csv("../input/mobile_train.csv")
    X = df.drop('price_range', axis=1).to_numpy()
    y= df.price_range.to_numpy()

    scl = preprocessing.StandardScaler()
    pca= decomposition.PCA()
    rf= ensemble.RandomForestClassifier(n_jobs=-1)

    classifier= pipeline.Pipeline(
            [
                ('scaling',  scl),
                ('pca', pca),
                ('rf', rf)
            ]
        )

    param_grid= {
        "pca__n_components": np.arange(5, 10),
        "rf__n_estimators": np.arange(100, 1500, 100), # for grid search: [100, 200, 300, 400], 
        "rf__max_depth": np.arange(1, 20),             # for grid search: [1, 3, 5, 7, 9, 11],
        "rf__criterion": ['gini', 'entropy']

    }

    model= model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter= 15,
        scoring="accuracy",
        verbose=10, 
        n_jobs= 1,
        cv=5
        )
    
    model.fit(X,y)

    print(model.best_score_)
    print(model.best_estimator_.get_params())