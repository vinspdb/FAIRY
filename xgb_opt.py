from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from hyperopt.pyll.base import scope
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class XGB():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.space =  {'learning_rate': hp.uniform("learning_rate", 0, 1),
                 'subsample': hp.uniform("subsample", 0.5, 1),
                 'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                 'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                 'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}

    def manual_cross_val_score(self, model, X, y, n_splits=3, random_state=42):
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Initialize scores array
        scores = np.zeros(n_splits)
        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            le = LabelEncoder()
            le.fit(list(y_train)+list(y_test))

            try:
                # Train the model
                model.fit(X_train, le.transform(y_train))

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate score
                score = accuracy_score(le.transform(y_test), le.inverse_transform(y_pred))
                scores[fold_idx] = score

            except Exception as e:
                print(f"Error in fold {fold_idx + 1}: {str(e)}")
                scores[fold_idx] = 0

        return np.mean(scores)

    def objective(self, params):

        model = XGBClassifier(n_estimators=500,
                                                learning_rate= params['learning_rate'],
                                                subsample=params['subsample'],
                                                max_depth=int(params['max_depth']),
                                                colsample_bytree=params['colsample_bytree'],
                                                min_child_weight=int(params['min_child_weight']),
                                                random_state=42
                                       )

        accuracy = self.manual_cross_val_score(model,self.X_train,self.y_train,3,42)
        return {'loss': -accuracy, 'status': STATUS_OK}

    def find_best(self):
        trials = Trials()
        best = fmin(fn=self.objective,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials,rstate=np.random.RandomState(42))
        return best
