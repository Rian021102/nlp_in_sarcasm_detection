import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import re
from bs4 import BeautifulSoup
import unicodedata
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

def datapipeline(pathfile):
    df = pd.read_csv(pathfile)
    df.drop_duplicates(subset="headline", keep='last', inplace=True)
    X = df[['headline']]  # Return a DataFrame with 'headline' column
    y = df['is_sarcastic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.head())
    return X_train, X_test, y_train, y_test

class TextPreprocessor:
    def __init__(self):
        self.stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        self.stop.update(punctuation)

    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def remove_between_square_brackets(self, text):
        return re.sub(r'\[[^]]*\]', '', text)

    def remove_urls(self, text):
        return re.sub(r'http\S+', '', text)

    def remove_stopwords(self, text):
        final_text = []
        for word in text.split():
            if word.strip().lower() not in self.stop:
                final_text.append(word.strip())
        return " ".join(final_text)

    def remove_accented_chars(self, text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def remove_punctuation(self, text):
        return re.sub(r'[^a-zA-Z0-9]', ' ', text)

    def remove_irrelevant_chars(self, text):
        return re.sub(r'[^a-zA-Z]', ' ', text)

    def remove_extra_whitespaces(self, text):
        return re.sub(r'^\s*|\s\s*', ' ', text).strip()
    
    def lemmatize_words(self,text):
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        words = [lemmatizer.lemmatize(word,pos='v') for word in words]
        return ' '.join(words)

    def preprocess_text(self, text):
        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_urls(text)
        text = self.remove_stopwords(text)
        text = self.remove_accented_chars(text)
        text = self.remove_punctuation(text)
        text = self.remove_irrelevant_chars(text)
        text = self.remove_extra_whitespaces(text)
        return text