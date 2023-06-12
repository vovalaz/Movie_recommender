import pandas as pd
import random
import math
import numpy as np


movies_file = "movies.csv"

movies = pd.read_csv(movies_file)

user_id_list = []
movie_id_list = []
rating_list = []

for i, movie in movies.iterrows():
    for j in range(random.randint(20, 40)):
        user_id = i
        movie_id = movie["movie_id"]
        a = math.floor(movie["vote_average"]) - 1 if math.floor(movie["vote_average"]) - 1 > 0 else 0
        b = math.ceil(movie["vote_average"]) + 1 if math.ceil(movie["vote_average"]) + 1 < 10 else 10
        rating = random.randint(a, b)
        user_id_list.append(user_id)
        movie_id_list.append(movie_id)
        rating_list.append(rating)
np.random.shuffle(user_id_list)
ratings = pd.DataFrame({"user_id": user_id_list, "movie_id": movie_id_list, "rating": rating_list})
ratings.drop_duplicates(subset=["user_id", "movie_id"])
ratings = ratings.sort_values(by="user_id")

ratings.to_csv("ratings.csv", index=False)
