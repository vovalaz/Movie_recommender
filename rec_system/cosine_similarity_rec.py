import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_json("movies.json")

# Create a count vectorizer object
count_vectorizer = CountVectorizer(stop_words="english")

# Preprocess the data
df["features"] = df["genres"] + " " + df["title"] + " " + str(df["vote_average"])
features_matrix = count_vectorizer.fit_transform(df["features"])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(features_matrix, features_matrix)


# Function to get movie recommendations based on cosine similarity scores
def get_recommendations(title, cosine_sim, df, top_n=5):
    # Get the index of the movie that matches the title
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with the given movie
    similarity_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top-n most similar movies
    top_movie_indices = [i[0] for i in similarity_scores[1:top_n+1]]

    # Return the top-n similar movies
    return df["title"].iloc[top_movie_indices]


# Example usage
movie_title = "The Dark Knight"
recommendations = get_recommendations(movie_title, cosine_sim, df)
print(f"Recommendations for \"{movie_title}\":")
print(recommendations)
