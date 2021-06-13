#Bag of words:

sentences=["One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The...","A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B...","I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con..."]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
_=vectorizer.fit_transform(sentences)

print(_)
print(vectorizer.vocabulary_)

#Tokenization

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sentences)
sequence=tokenizer.texts_to_sequences(sentences)
print(sequence)
print(tokenizer.word_index)

#Stop words removal(eg:the,a,in,etc)

from nltk.corpus import stopwords
s=set(stopwords.words("english"))
print(s)

for sen in sentences:
    print([word for word in sen.split() if word not in s])
    
import string
punc=set(string.punctuation)
print(punc)

#Stemming

#Playing-->play(correct)
#News-->New(wrong)

from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
stemmer.stem("News")


#Lemmatization

#Caring-->care(correct)

#But in Stemming-->caring-->car(wrong)

from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize("went")

import pandas as pd
import numpy as np
import tensorflow as tf

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# sequence=pad_sequences(sequence,padding="post",maxlen=20,truncating="post")
# print(sequence)

data=pd.read_csv("IMDB Dataset.csv")

x=data.iloc[:,0].values
y=data.iloc[:,1].values

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
y=lb.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer=Tokenizer(num_words=20000)

print(len(tokenizer.word_counts))


tokenizer.fit_on_texts(x_train)
x_train=tokenizer.texts_to_sequences(x_train)
x_test=tokenizer.texts_to_sequences(x_test)

x_train=pad_sequences(x_train,padding="post")
x_test=pad_sequences(x_test,padding="post")

model=tf.keras.Sequential([tf.keras.layers.Embedding(108943,64),tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(1)])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(1e-2),metrics=["accuracy"])

history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
