import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer




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

st.title('SPAM CLASSIFIER')
# take input
input_txt = st.text_input("Enter the message")
# loading the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

btn = st.button('Predict')
if btn:
    # preprocess the input text
    transformed_txt = transform(input_txt)
    vectorized_txt = tfidf.transform([transformed_txt])
    # feed it to model
    output = model.predict(vectorized_txt)[0]
    # output the result
    if output == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')