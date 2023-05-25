from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, ForeignKey, Float, Integer, Date, Text


Base = declarative_base()
metadata = Base.metadata


class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    message_text = Column(Text)


class User(Base):
    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String(255))


class Movie(Base):
    __tablename__ = "movie"
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    original_language = Column(String(5))
    popularity = Column(Float(5, True))
    release_date = Column(Date(), nullable=True)
    vote_average = Column(Float(7, True), nullable=True)
    genres = Column(Text, nullable=True)
    cast = Column(Text, nullable=True)
    directors = Column(Text, nullable=True)


class MovieUser(Base):
    __tablename__ = "movieuser"

    movie_id = Column(Integer, ForeignKey("movie.id"), nullable=False, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False, primary_key=True)
    rating = Column(Integer, nullable=False)
