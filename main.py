import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import string


def transform(text):
    # Lowercase the text
    text = text.lower()

    # Tokenize into words
    tokens = nltk.word_tokenize(text)

    # Initialize stemmer and stopword list
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Clean and stem
    cleaned = []
    for word in tokens:
        if word.isalnum() and word not in stop_words:
            stemmed_word = stemmer.stem(word)
            cleaned.append(stemmed_word)

    return " ".join(cleaned)

