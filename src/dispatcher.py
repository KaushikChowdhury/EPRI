from sklearn import ensemble
import lightgbm as lgb
from sklearn import linear_model
# from sklearn.ensemble import VotingRegressor

estimators_stacking = [('randomforest', ensemble.RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=2)),
                        ('extratrees', ensemble.ExtraTreesRegressor(n_estimators=200, n_jobs=-1, verbose=2))]


final_estimator = linear_model.LinearRegression()

MODELS = {
    "randomforest": ensemble.RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesRegressor(n_estimators=200, n_jobs=-1, verbose=2),
    "adaboost": ensemble.AdaBoostRegressor(n_estimators= 200),
    "lgb": lgb,
    "linear": linear_model.LinearRegression(),
    # "voting": VotingRegressor(),
    "stacking": ensemble.StackingRegressor(estimators = estimators_stacking, final_estimator = final_estimator, n_jobs = -1)
}