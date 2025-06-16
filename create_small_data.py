import os
import pandas as pd
import utils

logger = utils.set_logger (__name__)

# read in the data and add headers.
df_train = utils.load_data ('train.csv')
df_test = utils.load_data ('test.csv')

# Take 1% sample from df_train and df_test
df_train_small = df_train.sample(frac=0.0005, random_state=42)
df_test_small = df_test.sample(frac=0.0005, random_state=42)

# Save the sampled dataframes to new CSV files with '_small' appended to the new file names and keep in the same folder
utils.save_data(df_train_small, 'train_small.csv')
utils.save_data(df_test_small, 'test_small.csv')