import pickle as pkl 
import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
  



def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():   #alphanumeric
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf=pkl.load(open('Vectorizer.pkl','rb+'))
model=pkl.load(open('mnb.pkl','rb+'))

st.title('E-mail Spam Classifier')


input_sms=st.text_input("Enter the Message")
#steps
# 1.Preprocessing
# 2.Vectorizer
# 3.predict
# 4.display
if st.button('Predict'):

    transformed_sms=transform_text(input_sms)
    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]
    if result == 1:
        st.header('!!!Spam!!!')
    else:
        st.header('Not Spam')


