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


def get_movie_details(movie_id):
    # Construct the URL for the API request
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits"

    try:
        # Send the HTTP GET request
        response = requests.get(url)
        data = response.json()

        # Extract the cast and directors from the response
        if 'credits' in data:
            cast = data['credits']['cast']
            directors = [crew['name'] for crew in data['credits']['crew'] if crew['job'] == 'Director']

            return cast, directors
        else:
            print("Credits not found for the movie.")
    except requests.exceptions.RequestException as e:
        print("Error occurred during the API request:", e)


with open(filename, "a", encoding="utf-8") as f:
    f.write("[")
    for page in range(1, num_pages + 1):
        movies = []
        response = requests.get(url + str(page))
        data = response.json()

        if "results" in data:
            for movie in data["results"]:
                movie["genres"] = " ".join([genres[genre] for genre in movie["genre_ids"]])
                del movie["genre_ids"]
                cast, directors = get_movie_details(movie["id"])
                if cast:
                    cast_names = [actor["name"] for actor in cast]
                    movie["cast"] = ", ".join(cast_names)
                if directors:
                    movie["directors"] = ", ".join(directors)
                movies.append(movie)

        f.write(str(json.dumps(movies))[1:-1])
        if not page == num_pages:
            f.write(", ")
        print(f"Page {page} dumped")
    f.write("]")
