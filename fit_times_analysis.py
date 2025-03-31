import pandas as pd


#%%
fitting_times = pd.read_csv('./data/processed/fit_times.csv')

mean_times = fitting_times.groupby("MODEL")["FIT_TIME_SECONDS"].mean().sort_values().reset_index()
std_times = fitting_times.groupby("MODEL")["FIT_TIME_SECONDS"].std().sort_values().reset_index()

summary = fitting_times.groupby("MODEL")["FIT_TIME_SECONDS"].agg(['mean', 'std']).reset_index()
summary = summary.sort_values(by='mean')

