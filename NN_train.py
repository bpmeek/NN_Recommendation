#import dependencies
import numpy as np
import pandas as pd
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model
from sklearn.model_selection import train_test_split

#set debug to true for faster processing
debug = False

warnings.filterwarnings('ignore')

#load data
def get_data():
    if debug==True:
        dataset = pd.read_csv('ml-20m/ratings.csv', nrows=100)
    else:
        dataset = pd.read_csv('ml-20m/ratings.csv')
    return dataset

def process_data(dataset):
    #split data
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    return train, test

def build_model(dataset):
    #create counts for embeddings
    n_users = len(dataset.userId.unique())
    n_movies = len(dataset.movieId.unique())

    #create embedding path
    movie_input = Input(shape=[1], name="Movie-Input")
    movie_embedding = Embedding(n_movies+1, 5, name="Movie-Embedding")(movie_input)
    movie_vec = Flatten(name="Flatten-Movies")(movie_embedding)

    #create user embedding path
    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, 5, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    #concatenate features
    conc = Concatenate()([movie_vec, user_vec])

    #fully connected layers
    fc1 = Dense(128, activation='relu')(conc)
    fc2 = Dense(32, activation='relu')(fc1)
    out = Dense(1)(fc2)

    #create and compile model
    model = Model([user_input, movie_input], out)
    model.compile('adam','mean_squared_error')
    model.summary()

    return model

def train_model(model,train):
    if debug==True:
        epochs=1
        batch_size=100
    else:
        epochs=5
        batch_size=128

    model.fit([train.userId, train.movieId], train.rating, batch_size=128, epochs=epochs, verbose=1)
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())
    model.save('NN_Movie_Model.h5')
    return model

def eval_model(model,test):
    print(model.evaluate([test.userId, test.movieId], test.rating))

if __name__ == "__main__":
    dataset = get_data()
    train, test = process_data(dataset)
    model = build_model(dataset)
    trained_model = train_model(model,train)
    eval_model(trained_model,test)