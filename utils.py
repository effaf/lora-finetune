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
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def is_colab():
    """Detect if running in Google Colab."""
    try:
        return "google.colab" in str(get_ipython())
    except NameError:
        return False

def load_env():
    """
    Loads environment variables from .env.colab if running in Google Colab, otherwise from .env.local.
    """
    from pathlib import Path
    from dotenv import load_dotenv

    env_file = '.env.colab' if is_colab() else '.env.local'
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.warning(f"Environment file {env_file} not found.")

def init(logger_name):
    logger= set_logger(logger_name)
    load_env()
    return logger


def get_data_dir ():
    return os.getenv ('DATA_DIR', r"C:\Users\shlok\.cache\kagglehub\datasets\kritanjalijain\amazon-reviews\versions\2")

def save_data (df, file_name):
    df.to_csv(os.path.join(get_data_dir(), file_name), index=False)
    logger.info(f"Saved:{file_name}: {df.shape}")

def load_data (file_name):
    df = pd.read_csv(os.path.join(get_data_dir(), file_name), header=None, names=['polarity', 'title', 'text'])
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