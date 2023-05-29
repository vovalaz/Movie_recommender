import telebot
import random

from tg_bot.bot import bot
from tg_bot import markups
from tg_bot.states import TestStates
from db import orm


@bot.message_handler(state=TestStates.test)
async def test_handler(message: telebot.types.Message):
    if message.text == "/end":
        pass
    elif message.text == "/skip":
        movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
        movie_id = orm.get_movie_by_title(movie_title).movie_id
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        orm.create_movie_user(movie_id, user_id, None)
    elif message.text == "/start_test":
        movie = orm.get_movie(random.randint(1, 500))
        for movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            print(f"Rating = {movie.rating}")
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))
        await bot.send_message(
            message.chat.id,
            "Rate this film:",
            reply_markup=markups.rating_markup(),
        )
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    elif message.text.isnumeric():
        pass
