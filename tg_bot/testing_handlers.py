import random
import telebot

from db import orm
from tg_bot.bot import bot
from tg_bot import markups
from tg_bot.states import TestStates


@bot.message_handler(state=TestStates.test)
async def test_handler(message: telebot.types.Message):
    if message.text == "/start_test":
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Rate this film:",
        )
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    elif message.text == "/end":
        await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
        await bot.reply_to(message, "Ratings saved.\nWhat you want to do next?", reply_markup=markups.rec_markup())

    elif message.text == "/dont_recommend":
        movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
        movie_id = orm.get_movie_by_title(movie_title).movie_id
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        orm.create_or_update_movie_user(movie_id, user_id, None)

        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Previous film will not be suggested in tests anymore.\nRate this film:",
        )
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    elif message.text == "/skip":
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Skipped previous film.\nRate this film:",
        )
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    elif message.text.isnumeric():
        rating = int(message.text)
        if 0 <= rating <= 10:
            movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
            movie_id = orm.get_movie_by_title(movie_title).movie_id
            user_id = orm.get_user_by_username(message.from_user.username).user_id
            orm.create_or_update_movie_user(movie_id, user_id, int(message.text))

            movie = orm.get_movie(random.randint(1, 500))
            while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
                movie = orm.get_movie(random.randint(1, 500))

            await bot.send_message(
                message.chat.id,
                "Rate this film:",
            )
            sent_movie_title = await bot.send_message(
                message.chat.id,
                str(movie.title),
                reply_markup=markups.rating_markup(),
            )
            await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_movie_title.text,
                sent_movie_title.chat.id,
                True,
            )
        else:
            await bot.send_message(
                message.chat.id,
                "Rating you provide should be in [0, 10] range to be processed properly\nRate this film again:",
            )
            movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
            sent_movie_title = await bot.send_message(
                message.chat.id,
                str(movie_title),
                reply_markup=markups.rating_markup(),
            )
            await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie_title)))
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_movie_title.text,
                sent_movie_title.chat.id,
                True,
            )
