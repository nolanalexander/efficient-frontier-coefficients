import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression

'''
Defines a ForecastModel object used for define more complex forecasting
models that using multiple estimator models. Also ensures that the 
model has the same methods (fit, tune, forecast), so there is no 
conflict between syntax for estimator packages (e.g. Scikit-learn and Statsmodels)
'''

# Abstract Class to be extended
class ForecastModel:
    
    def __init__(self):
        self.models = [None]
        self.has_hyperparams = None
        self.is_continuous = None

    def fit(self, X, y):
        raise NotImplementedError("train() not implemented")
        
    def tune(self, X, y, hyperparams_dict):
        raise NotImplementedError("tune() not implemented")
    
    def forecast(self, X):
        raise NotImplementedError("forecast() not implemented")

class RFForecastModel(ForecastModel):
    def __init__(self):
        self.models = [RandomForestClassifier(random_state=0)]
        self.has_hyperparams = True
        self.is_continuous = False
        
    def fit(self, X, y):
        self.models[0] = self.models[0].fit(X, y)
        return self
    
    def tune(self, X, y, param_grid, n_splits, scoring):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        grid = GridSearchCV(estimator=self.models[0], cv=tscv, param_grid=param_grid, scoring=scoring)
        grid.fit(X, y)
        self.models[0] = grid.best_estimator_
        return self, grid.best_params_
        
    def forecast(self, X, return_proba=True):
        return self.models[0].predict_proba(X)[:, 1] if return_proba else self.models[0].predict(X)
    
class LassoForecastModel(ForecastModel):
    def __init__(self):
        self.models = [Lasso(random_state=0), LinearRegression()]
        self.has_hyperparams = True
        self.is_continuous = True
        
    def fit(self, X, y):
        self.models[0] = self.models[0].fit(X, y)
        return self
    
    def tune(self, X, y, param_grid, n_splits, scoring):
        if self.models[0] == LinearRegression():
            self.models[0] = Lasso(random_state=0)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        param_grid = param_grid.copy()
        if 'alpha' in param_grid:
            param_grid['alpha'] = np.array(param_grid['alpha'])[~np.isclose(param_grid['alpha'], 0, rtol=1e-20, atol=1e-20)]
        grid = GridSearchCV(estimator=self.models[0], cv=tscv, param_grid=param_grid, scoring=scoring)
        grid.fit(X, y)
        self.models[0] = grid.best_estimator_
        best_params = grid.best_params_
        if 'alpha' in param_grid and (np.isclose(param_grid['alpha'], 0, rtol=1e-20, atol=1e-20).any() and 
            cross_val_score(LinearRegression(), X, y, scoring=scoring) > cross_val_score(self.models[0], X, y, scoring=scoring)):
            self.models[0] = LinearRegression()
            best_params['alpha'] = 0
        return self, best_params
        
    def forecast(self, X, return_proba=False):
        return self.models[0].predict(X)
    
    
    
        