import pandas as pd
import random
import string

from db.orm import (
    create_movie,
    get_movie,
)


def generate_username():
    # Generate a random username of length 8
    length = random.randint(7, 12)
    letters = string.ascii_lowercase
    username = "".join(random.choice(letters) for _ in range(length))

    return username


def init_db():
    movies = pd.read_csv("movies.csv")
    if not get_movie(0):
        create_movie(
            title="abc",
            id=0,
            original_language="br",
            popularity=0.001,
            release_date="1999-12-01",
            vote_average=5.0,
            genres="abc",
            cast="abc",
            directors="abc",
        )
    for index, row in movies.iterrows():
        if not get_movie(index + 1):
            create_movie(
                title=row["title"],
                id=row["id"],
                original_language=row["original_language"],
                popularity=row["popularity"],
                release_date=row["release_date"],
                vote_average=row["vote_average"],
                genres=row["genres"],
                cast=row["cast"],
                directors=row["directors"],
            )


if __name__ == "__main__":
    init_db()
