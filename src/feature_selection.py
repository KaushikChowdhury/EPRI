
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from sklearn import feature_selection
import joblib
from . import feature_generation
from . import dispatcher
from . import constants
import lightgbm as lgb

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TESTING_DATA = os.environ.get("TESTING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

FOLD_MAPPPING = {
    0: [-1],
    1:  [-1, 0],
    2:  [-1, 0, 1],
    3:  [-1, 0, 1, 2],
    4:  [-1, 0, 1, 2, 3],
    5:  [-1, 0, 1, 2, 3, 4],
    6:  [-1, 0, 1, 2, 3, 4, 5],
    7:  [-1, 0, 1, 2, 3, 4, 5, 6],
    8:  [-1, 0, 1, 2, 3, 4, 5, 6, 7],
    9:  [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA, parse_dates = ['timestampUTC'], infer_datetime_format=True,na_values=['nan','?'] ,index_col='timestampUTC')
    df.drop(columns = df.columns[0], inplace = True )
    df["timestamp"] = df.index
    
    
    
    test_df = pd.read_csv(TESTING_DATA, parse_dates = ['timestampUTC'], infer_datetime_format=True,na_values=['nan','?'] ,index_col='timestampUTC')
    test_df["timestamp"] = test_df.index
    
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    
    train_df = feature_generation.get_time_features(train_df, time_zone = constants.site_tz)
    train_df = feature_generation.f_create_lag_features(train_df)
    train_df.dropna(subset= [col for col in train_df.columns if col not in "measured_kW"] , inplace= True)
    train_df.dropna(inplace = True)

    valid_df = feature_generation.get_time_features(valid_df, time_zone = constants.site_tz)
    valid_df = feature_generation.f_create_lag_features(valid_df)
    valid_df.dropna(subset= [col for col in valid_df.columns if col not in "measured_kW"] , inplace= True)
    valid_df.dropna(inplace = True)
    
    
    test_df = feature_generation.get_time_features(test_df, time_zone = constants.site_tz)
    test_df = feature_generation.f_create_lag_features(test_df)
    test_df.dropna(subset= [col for col in test_df.columns if col not in "measured_kW"] , inplace= True)
    test_df.dropna(inplace = True)
    ytrain = train_df.measured_kW.values
    yvalid = valid_df.measured_kW.values
    ytest = test_df.measured_kW.values
    
    
    train_df = train_df.drop(columns = ["measured_kW", "timestamp", "kfold"]).reset_index(drop = True)
    valid_df = valid_df.drop(columns = ["measured_kW", "timestamp", "kfold"]).reset_index(drop = True)
    test_df = test_df.drop(columns = ["measured_kW", "timestamp"]).reset_index(drop = True)

    valid_df = valid_df[train_df.columns]
    test_df = test_df[train_df.columns]

    categorical_columns = ['weekday', 'month', 'day', 'hour', 'is_weekend', 'is_start_of_week']
    # print(train_df.head())


    label_encoders = {}
    for c in categorical_columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist()+ test_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders[c] = lbl

    clf = dispatcher.MODELS[MODEL]

    model_pipeline = Pipeline(steps = [
        ("regressor", clf)
    ])

    if MODEL == "lgb":
        params = {
                             'boosting_type':'gbdt',
                             'objective':'regression',
                             'metric':'rmsle',
                             'learning_rate':0.0001,
                             'verbose':1
                             }
        n_estimators = 5000
        d_train=lgb.Dataset(train_df,label=ytrain)
        clf = lgb.train(params,d_train,n_estimators,verbose_eval=1)
        preds = clf.predict(valid_df)

    elif MODEL == "stacking":
        clf.fit(train_df, ytrain)
        preds = clf.predict(valid_df)

    else:
        model_pipeline.fit(train_df, ytrain)
        preds = clf.predict(valid_df)
    
    
    print(metrics.mean_absolute_error(yvalid, preds))
    joblib.dump(label_encoders, f"models/{MODEL}_{FOLD}_label_encoder.pkl")
    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")
    pd.DataFrame({"fold" : FOLD, "metric" : metrics.mean_absolute_error(yvalid, preds)}, columns=["fold", "metric"], index = [0,1]).to_csv(f"models/{MODEL}_{FOLD}_metric.csv", index=False)
