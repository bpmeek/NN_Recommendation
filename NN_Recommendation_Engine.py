#import dependencies
import sys
import pandas as pd
import numpy as np
from keras.models import model_from_json

if len(sys.argv) != 2:
    print("Usage: python NN_Recommendation_Engine.py [userId]")
    exit()

userId = int(sys.argv[1])

def get_data():
    dataset = pd.read_csv('ml-20m/ratings.csv')
    return dataset

def get_model():
    with open('model_architecture.json','r') as f:
        model = model_from_json(f.read())
    model.load_weights('NN_Movie_Model.h5')
    return model

def get_user_data(userId,dataset):
    #data set for given user
    movie_data = np.array(list(set(dataset.movieId)))
    user = np.array([1 for i in range(len(movie_data))])
    return movie_data, user

def predict(model, movie_data, user):
    #predict movies for given user
    predictions = model.predict([user, movie_data])
    predictions = np.array([a[0] for a in predictions])
    #keep top 5 predictions
    recommended_movie_ids = (-predictions).argsort()[:5]
    return recommended_movie_ids

if __name__ == "__main__":
    dataset = get_data()
    movie_data, user = get_user_data(userId,dataset)
    model = get_model()
    recommendations = predict(model, movie_data, user)
    print(recommendations)
