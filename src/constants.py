import os


TEST_CREATION_DATE = "2020-01-01 00:00:00"
cont_mode = 0
site_tz= 'US/Pacific'
num_lags_disc=[24*2, 24*3, 24*4, 24*5,24*6, 24*7,24*8,24*9]
n_timesteps=num_lags_disc[0]
forecast_window = ["DA"]#["HASHA", "DA", "FDA"]
combineHistYN =  True