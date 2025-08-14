import numpy as np
import pandas as pd
import os
import re
import nltk
from typing import Any
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

import logging

#logging config
logger = logging.getLogger('data_processing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
  """Lemmatize the text"""
  lemmatizer = WordNetLemmatizer()
  text = text.split()
  text = [lemmatizer.lemmatize(y) for y in text]
  return " ".join(text)

def remove_stopwords(text):
  """Remove stop words from the text."""
  stop_words = set(stopwords.words('english'))
  Text = [i for i in str(text).split() if i not in stop_words]
  return " ".join(Text)

def removing_numbers(text):
  """Remove numbers from the text"""
  text = ''.join([i for i in text if not i.isdigit()])
  return text

def lower_case(text):
  """Convert text to lower case."""
  text = text.split()
  text = [y.lower() for y in text]
  return " ".join(text)

def removing_punctuations(text):
  """##Remove punctuations from the text"""
  text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
  text = text.replace('؛',"", )
  ##remove extra white space
  text = re.sub('\s+', ' ',text)
  text = " ".join(text.split())
  return text.strip()

def remove_urls(text):
  """Remove URLs from the text."""
  url_pattern = re.compile(r'http?://\S+|www\.\S+')
  return url_pattern.sub(r'', text)


def remove_small_stentences(df):
  """Remove sentences with less than 3 words"""
  for i in range(len(df)):
    if len(df.text.iloc[i].split()) <3:
      df.text.iloc[i] = np.nan

def normalize_text(df):
  """Normalize the text data."""
  try:
    df['content'] = df['content'].apply(lower_case)
    logger.debug('Converted to lower case')
    df['content'] = df['content'].apply(remove_stopwords)
    logger.debug('Stop words removed')
    df['content'] = df['content'].apply(removing_numbers)
    logger.debug('numbers removed')
    df['content'] = df['content'].apply(removing_punctuations)
    logger.debug('punctuations removed')
    df['content'] = df['content'].apply(remove_urls)
    logger.debug('urls')
    df['content'] = df['content'].apply(lemmatization)
    logger.debug('Lemmatization performed')
    logger.debug('Text Normalization completed')
    return df
  except Exception as e:
    logger.error('Error during text normalization: %s', e)
    raise
  
def main():
  try:
    #Fetch the data from data/raw
    train_data = pd.read_csv('./data/raw/train.csv')
    test_data = pd.read_csv('./data/raw/test.csv')
    logger.debug('data loaded properly')

    #Transform the data
    train_processed_data = normalize_text(train_data)
    test_processed_data  = normalize_text(test_data)

    #Store the data inside data/process
    data_path = os.path.join("./data", "preprocessed")
    os.makedirs(data_path, exist_ok=True)

    train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
    test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

    logger.debug('Processed data saved to %s', data_path)


  except Exception as e:
    logger.error('Failed to complete the data transformation process: %s', e)
    print(f"Error: {e}")

if __name__ == '__main__':
  main()