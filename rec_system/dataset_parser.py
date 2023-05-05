import json
import requests
import os

api_key = "9933450629ff7370fbe8df044670c3a0"
url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={api_key}&page="
num_pages = 500
filename = "movies.json"
genres_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={api_key}"
genres = requests.get(genres_url).json()["genres"]
genres = {genre["id"]: genre["name"] for genre in genres}

if os.path.exists(filename):
    os.remove(filename)

with open(filename, "a", encoding="utf-8") as f:
    f.write("[")
    for page in range(1, num_pages + 1):
        movies = []
        response = requests.get(url + str(page))
        data = response.json()

        if "results" in data:
            for movie in data["results"]:
                movie["genres"] = [genres[genre] for genre in movie["genre_ids"]]
                del movie["genre_ids"]
                movies.append(movie)

        f.write(str(json.dumps(movies))[1:-1])
        if not page == num_pages:
            f.write(", ")
        print(f"Page {page} dumped")
    f.write("]")
