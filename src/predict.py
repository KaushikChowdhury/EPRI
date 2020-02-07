import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np
from . import constants
from . import dispatcher
from . import feature_generation

TESTING_DATA = os.environ.get("TESTING_DATA")
MODEL = os.environ.get("MODEL")


def predict():
    df = pd.read_csv(TESTING_DATA, parse_dates = ['timestampUTC'], infer_datetime_format=True,na_values=['nan','?'] ,index_col='timestampUTC')
    df["timestamp"] = df.index
    predictions = None

    for FOLD in range(10):
        print(FOLD)
        df = pd.read_csv(TESTING_DATA, parse_dates = ['timestampUTC'], infer_datetime_format=True,na_values=['nan','?'] ,index_col='timestampUTC')
        df["timestamp"] = df.index
        
        
        df = feature_generation.get_time_features(df, time_zone = constants.site_tz)
        df = feature_generation.f_create_lag_features(df)
        df.dropna(subset= [col for col in df.columns if col not in "measured_kW"] , inplace= True)
        test_idx = df.timestamp.values
        

        ytest = df.measured_kW.values
        df = df.drop(columns = ["measured_kW", "timestamp"]).reset_index(drop = True)
        # print(ytest)
        


        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        
        # data is ready to train
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        
        df = df[cols]
        
        preds = clf.predict(df)

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 10

    sub = pd.DataFrame({"timestampUTC": test_idx, "actual_measured_kW":ytest , "predicted_measured_kW":predictions}, columns=["timestampUTC", "actual_measured_kW" , "predicted_measured_kW"])
    return sub, ytest, predictions
    

if __name__ == "__main__":
    submission, actual ,predictions = predict()
    print(metrics.mean_absolute_error(actual, predictions))
    submission.to_csv(f"models/{MODEL}.csv", index=False)