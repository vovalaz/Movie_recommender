import pandas as pd


def fix_movies(filename):
    df = pd.read_json("movies.json")
    # df = pd.read_csv(filename)
    df.insert(0, "movie_id", range(1, len(df) + 1))
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    fix_movies("movies.csv")
