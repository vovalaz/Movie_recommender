from sqlalchemy.orm import sessionmaker
from db.models import Message, MovieUser, Movie, UserInfo
from db.database import engine

Session = sessionmaker(bind=engine)
session = Session()


def list_movies_rated_by_user(user_id: int):
    movies = session.query(Movie).join(MovieUser).filter(MovieUser.user_id == user_id).all()
    return movies


def create_message(user_id, message_text, chat_id, from_bot=False):
    new_message = Message(user_id=user_id, message_text=message_text, chat_id=chat_id, from_bot=from_bot)
    session.add(new_message)
    session.commit()
    return new_message


def create_user(username):
    user = UserInfo(username=username)
    session.add(user)
    session.commit()
    return user


def create_user_if_not_exists(username: str) -> UserInfo:
    user = session.query(UserInfo).filter_by(username=username).first()
    if user is None:
        new_user = UserInfo(username=username)
        session.add(new_user)
        session.commit()
        return new_user
    return user


def create_movie(title, id, original_language, popularity, release_date, vote_average, genres, cast, directors):
    movie = Movie(
        title=title,
        id=id,
        original_language=original_language,
        popularity=popularity,
        release_date=release_date,
        vote_average=vote_average,
        genres=genres,
        cast=cast,
        directors=directors,
    )
    session.add(movie)
    session.commit()
    return movie


def create_movie_user(movie_id, user_id, rating):
    movie_user = MovieUser(movie_id=movie_id, user_id=user_id, rating=rating)
    session.add(movie_user)
    session.commit()
    return movie_user


def get_message(message_id):
    return session.query(Message).get(message_id)


def get_user(user_id):
    return session.query(UserInfo).get(user_id)


def get_user_by_username(username):
    return session.query(UserInfo).filter_by(username=username).first()


def get_movie(movie_id):
    return session.query(Movie).get(movie_id)


def get_movie_user(movie_id, user_id):
    return session.query(MovieUser).filter_by(movie_id=movie_id, user_id=user_id).first()


def get_last_message_by_chat_id(chat_id):
    last_message = session.query(Message).filter(Message.chat_id == chat_id).order_by(Message.id.desc()).first()
    return last_message


def get_movie_by_title(title):
    movie = session.query(Movie).filter(Movie.title == title).first()
    return movie


def create_or_update_movie_user(movie_id, user_id, rating):
    movie_user = session.query(MovieUser).filter(MovieUser.movie_id == movie_id, MovieUser.user_id == user_id).first()

    if movie_user:
        movie_user.rating = rating
    else:
        movie_user = MovieUser(movie_id=movie_id, user_id=user_id, rating=rating)
        session.add(movie_user)

    session.commit()
    return movie_user


def update_message(message_id, new_message_text):
    message = get_message(message_id)
    if message:
        message.message_text = new_message_text
        session.commit()
        return message
    else:
        raise ValueError("Message not found.")


def update_user(user_id, new_username):
    user = get_user(user_id)
    if user:
        user.username = new_username
        session.commit()
        return user
    else:
        raise ValueError("User not found.")


def update_movie(
    movie_id,
    new_title,
    new_original_language,
    new_popularity,
    new_release_date,
    new_vote_average,
    new_genres,
    new_cast,
    new_directors,
):
    movie = get_movie(movie_id)
    if movie:
        movie.title = new_title
        movie.original_language = new_original_language
        movie.popularity = new_popularity
        movie.release_date = new_release_date
        movie.vote_average = new_vote_average
        movie.genres = new_genres
        movie.cast = new_cast
        movie.directors = new_directors
        session.commit()
        return movie
    else:
        raise ValueError("Movie not found.")


def update_movie_user(movie_id, user_id, new_rating):
    movie_user = get_movie_user(movie_id, user_id)
    if movie_user:
        movie_user.rating = new_rating
        session.commit()
        return movie_user
    else:
        raise ValueError("MovieUser not found.")


def delete_message(message_id):
    message = get_message(message_id)
    if message:
        session.delete(message)
        session.commit()
        return True
    else:
        raise ValueError("Message not found.")


def delete_user(user_id):
    user = get_user(user_id)
    if user:
        session.delete(user)
        session.commit()
        return True
    else:
        raise ValueError("User not found.")


def delete_movie(movie_id):
    movie = get_movie(movie_id)
    if movie:
        session.delete(movie)
        session.commit()
        return True
    else:
        raise ValueError("Movie not found.")


def delete_movie_user(movie_id, user_id):
    movie_user = get_movie_user(movie_id, user_id)
    if movie_user:
        session.delete(movie_user)
        session.commit()
        return True
    else:
        raise ValueError("MovieUser not found.")
