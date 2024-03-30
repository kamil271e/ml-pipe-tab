import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


# Should I consider using optuna here or it is overkill?


def CVregressor(cv, rand, regressor, params):
    if rand:
        search = RandomizedSearchCV(regressor, param_distributions=params, cv=cv)
    else:
        search = GridSearchCV(regressor, param_grid=params, cv=cv)
    return search


def stacking_regressor_pipe(X, y, regressors, params, final_estimator):
    ready_reg = []
    for reg, p in zip(regressors, params):
        ready_reg.append(CVregressor(cv=10, rand=True, regressor=reg, params=p))

    for i in range(len(ready_reg)):
        ready_reg[i].fit(X, y)

    estimators = [(str(reg), reg.best_estimator_) for reg in ready_reg]
    ensemble = StackingRegressor(estimators, final_estimator=final_estimator)
    ensemble.fit(X, y)

    return ensemble, estimators


def plot_weak_learners_feature_importance(feature_importances, nlargest=10):
    for i in range(len(feature_importances)):
        d = feature_importances.iloc[i][feature_importances.iloc[i] > 0]
        plt.figure(figsize=(5, 5))
        pd.Series(
            feature_importances.iloc[i], index=feature_importances.columns
        ).nlargest(nlargest).plot(kind="barh")
        plt.show()
