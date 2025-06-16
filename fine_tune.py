import os
import utils
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from wordcloud import WordCloud, STOPWORDS
import warnings

logger = utils.init (__name__)

# Check if GPU is available and set the device accordingly
if torch.cuda.is_available():
    logger.info("GPU is enabled and available for use.")
    device = 0  # GPU device index
else:
    logger.info("GPU is not available, running on CPU.")
    device = -1  # CPU

# Load the sentiment analysis pipeline with GPU enabled
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=32, device=device)

# read in the data and add headers.
df_train = utils.load_data('train_small.csv')
df_test = utils.load_data('test_small.csv')

combined_df = utils.combine_dfs(df_train, df_test)
if utils.is_colab():
    subset_df = combined_df
else:
    subset_df = combined_df.sample(frac=1 random_state=42) 

# Run sentiment analysis on the subset
texts = subset_df['text'].tolist()
logger.info ("doing sentiments:%s", subset_df.shape)
results = sentiment_pipeline(texts, batch_size=32) 

# Extract sentiment labels and confidence scores
subset_df['Sentiment'] = [result['label'] for result in results]
subset_df['Confidence'] = [result['score'] for result in results]

logger.info ("sentiments:\n%s", subset_df.head(5))

# Compare polarity and sentiment columns
subset_df['Polarity_Sentiment_Match'] = subset_df['polarity'] == subset_df['Sentiment'].apply(lambda x: '1' if x == 'NEGATIVE' else '2')

polarity_sentiment_match = subset_df['Polarity_Sentiment_Match'].mean()

# Display match percentage
logger.info (f"Polarity matches sentiment: {polarity_sentiment_match * 100:.2f}%")
utils.save_data (subset_df, "small_with_sentiment.csv")