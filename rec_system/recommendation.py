import pandas as pd
from sklearn.neighbors import NearestNeighbors


df = pd.read_json("movies.json")

print(df)
