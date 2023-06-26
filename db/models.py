from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, ForeignKey, Float, Integer, Date, Text, Boolean

# Base model
Base = declarative_base()
metadata = Base.metadata


# Message model
class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("userinfo.user_id"))
    message_text = Column(Text)
    chat_id = Column(Integer)
    from_bot = Column(Boolean)


# User model
class UserInfo(Base):
    __tablename__ = "userinfo"

    user_id = Column(Integer, primary_key=True)
    username = Column(String(255))


# Movie model
class Movie(Base):
    __tablename__ = "movie"
    movie_id = Column(Integer, primary_key=True)
    title = Column(String(255))
    id = Column(String(255))
    original_language = Column(String(5))
    popularity = Column(Float(5, True))
    release_date = Column(Date(), nullable=True)
    vote_average = Column(Float(7, True), nullable=True)
    genres = Column(Text, nullable=True)
    cast = Column(Text, nullable=True)
    directors = Column(Text, nullable=True)


# Many to many movie-user relation table
class MovieUser(Base):
    __tablename__ = "movieuser"

    movie_id = Column(Integer, ForeignKey("movie.movie_id"), nullable=False, primary_key=True)
    user_id = Column(Integer, ForeignKey("userinfo.user_id"), nullable=False, primary_key=True)
    rating = Column(Integer, nullable=True)
