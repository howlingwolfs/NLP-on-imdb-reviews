import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
port_stemmer = PorterStemmer()

from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


class Howlingwolfs:
    def __init__(self):
        self.data = pd.read_csv('Train.csv')
        self.train = self.data.copy()
        self.train.drop_duplicates(inplace=True)
        self.train['tokens'] = self.train['text'].apply(word_tokenize)
        self.imdb = models.load_model('Imdb.keras')
        self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(self.train['tokens'])

    @staticmethod
    def preprocessing(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'[^\w\s,.!?]', '', text)  # removes most emojis & symbols
        text = text.split()
        text = [port_stemmer.stem(word) for word in text if not word in stop_words]
        return text

    def process_review(self, review, max_length=180):
        review_processed = self.preprocessing(review)
        review_token = word_tokenize(review)
        review_sequences = self.tokenizer.texts_to_sequences([review_token])
        review_final = pad_sequences(review_sequences, maxlen=max_length, padding='post', truncating='post')
        print(review_final.shape)
        return review_final

    def predict(self, review):
        predicting = self.imdb.predict(review, verbose=1)
        predicted_classes = (predicting > 0.5).astype("int32")
        predicted_classes.resize(1)
        ans = int(predicted_classes)
        if ans == 1:
            print('\n===< The review is positive >===\n')
        elif ans == 0:
            print('\n===< The review is negative >===\n')
        return None