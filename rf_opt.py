from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold

class RF():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.space =  {'max_features': hp.uniform('max_features', 0, 1)}


    def objective(self, params):
        model = RandomForestClassifier(max_features=params['max_features'],
                                       n_estimators=500, random_state=42
                                       )
        skf = StratifiedKFold(n_splits=3)

        accuracy = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring='f1_macro').mean()

        # We aim to maximize accuracy, therefore we return it as a negative value
        return {'loss': -accuracy, 'status': STATUS_OK}

    def find_best(self):
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials,rstate=np.random.RandomState(42))
        return best
