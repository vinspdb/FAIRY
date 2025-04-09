from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold


class LR():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.space = {'C': hp.uniform('C', -15, 15)}
    def objective(self, params):
        dist = (np.unique(self.y_train, return_counts=True))
        class_weight = {0: ((len(self.y_train)) / (2 * dist[1][0])), 1: ((len(self.y_train)) / (2 * dist[1][1]))}
        skf = StratifiedKFold(n_splits=3)
        model = LogisticRegression(C=2**params['C'], random_state=42, class_weight=class_weight)
        accuracy = cross_val_score(model, self.X_train, self.y_train, scoring='f1_macro', cv=skf).mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK}

    def find_best(self):
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials, rstate=np.random.RandomState(42))
        return best
