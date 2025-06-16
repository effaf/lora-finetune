import os
import logging
import pandas as pd

logger = None

def set_logger(logger_name):
    # Set up logger
    global logger
    logger = logging.getLogger(logger_name)
    # Remove all handlers associated with the logger
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

path_to_datasets = r"C:\Users\shlok\.cache\kagglehub\datasets\kritanjalijain\amazon-reviews\versions\2"
def save_data (df, file_name):
    df.to_csv(os.path.join(path_to_datasets, file_name), index=False)
    logger.info(f"Saved:{file_name}: {df.shape}")

def load_data (file_name):
    df = pd.read_csv(os.path.join(path_to_datasets, file_name), header=None, names=['polarity', 'title', 'text'])
    logger.info(f"Loaded:{file_name}: {df.shape}")

    return df

def combine_dfs (first, second):
    # append both files
    combined_df = pd.concat([first, second], ignore_index=True)

    # drop duplicates just in case
    combined_df.drop_duplicates(inplace=True)

    # show the structure of the dataframe
    logger.info(f"combined dfs: {first.shape} + {second.shape} = {combined_df.shape}")

    return combined_df