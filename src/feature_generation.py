from . import constants
import pytz
import datetime as dt

def f_create_lag_features(data,target_col="measured_kW", nlag= constants.num_lags_disc,cont_mode = constants.cont_mode):
    """This function is used to lags of time series varaible as added feature to input data
    """
    if(cont_mode==1):
        df = data
        for i in range(24*2,nlag+1):
            col=target_col+"_lag_"+str(i)
            df[col]=df[target_col].shift(i).values
    elif(cont_mode==0):
        df = data
        for i in nlag:
            col=target_col+"_lag_"+str(i)
            df[col]=df[target_col].shift(i).values
    return df

def get_time_features(df,time_zone):
    """ Returns the different features extracted from the specified date.
    """
    #####localtimezone
    lc_tz=pytz.timezone(time_zone)
    df['timestamp_local']= df.index.tz_localize(pytz.utc).tz_convert(lc_tz)
    
  
    
#    df['date']=df['timestamp_local'].apply(lambda x:x.date())
    df['weekday']=df['timestamp_local'].apply(lambda x:x.weekday()).values.tolist()  # Monday is 0, Sunday is 6.
    # df['year']=df['timestamp_local'].apply(lambda x:x.year).values.tolist()
    df['month']=df['timestamp_local'].apply(lambda x:x.month).values.tolist()
    ###df['year_month']=df['timestamp_local'].apply(lambda x: 100*x.year+x.month)
    df['day']=df['timestamp_local'].apply(lambda x:x.day).values.tolist()
    df['hour']=df['timestamp_local'].apply(lambda x:x.hour).values.tolist()
    
       
    # df['week_of_month'] = df['timestamp_local'].apply(lambda x:week_of_month(x))
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x==5 or x==6 else 0)
    df['is_start_of_week'] = df['weekday'].apply(lambda x: 1 if x==0 or x==1 else 0)
    
    ######

    
    df.drop('timestamp_local',axis=1,inplace=True)
    
    return df
