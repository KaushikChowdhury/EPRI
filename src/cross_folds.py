import pandas as pd
from sklearn import model_selection
from . import constants

if __name__ == "__main__":

    df = pd.read_csv("./input/Axum_clean_cont_RH_ST_combined.csv", parse_dates = ['timestampUTC'], infer_datetime_format=True,na_values=['nan','?'] ,index_col='timestampUTC')
    if constants.combineHistYN:
        df_hist = pd.read_csv("./input/Axum_clean_cont_Hist_combined.csv", parse_dates = ['timestampUTC'], infer_datetime_format=True,na_values=['nan','?'] ,index_col='timestampUTC')
        df = pd.concat([df_hist, df])
        df.dropna(inplace = True)
    else:
        pass
    # checking for forecating window and resampling the data
    if constants.forecast_window == ["DA"]:
        df = df.resample('60T').mean()
        df["kfold"] = -1
        df.dropna(inplace = True)
        test = df[df.index >= constants.TEST_CREATION_DATE]
        test.drop(columns = ["kfold"], inplace = True)
        df = df[df.index < constants.TEST_CREATION_DATE]
    else:
        df["kfold"] = -1
        df.dropna(inplace = True)
        test = df[df.index >= constants.TEST_CREATION_DATE]
        test.drop(columns = ["kfold"], inplace = True)
        df = df[df.index < constants.TEST_CREATION_DATE]

    # saving the file and reading it back so that the index in resetted.
    df.to_csv(f"input/train_folds_{constants.forecast_window}.csv", index=True)
    df = pd.read_csv(f"input/train_folds_{constants.forecast_window}.csv")
    
    # creating Time - based split
    tss = model_selection.TimeSeriesSplit(n_splits= 10)

    for fold, (train_idx, val_idx) in enumerate(tss.split(X = df.index)):
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv("input/train_folds.csv", index=True)
    test.to_csv("input/test.csv", index = True)
