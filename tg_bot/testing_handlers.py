import random
import telebot

from db import orm
from tg_bot.bot import bot
from tg_bot import markups
from tg_bot.states import TestStates


@bot.message_handler(state=TestStates.test)
async def test_handler(message: telebot.types.Message):
    if message.text == "/почати_опитування":
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Оцініть цей фільм:",
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
    elif message.text == "/закінчити":
        await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
        await bot.reply_to(message, "Рейтинги збережено.\nОберіть наступну дію", reply_markup=markups.rec_markup())

    elif message.text == "/не_рекомендувати":
        movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
        movie_id = orm.get_movie_by_title(movie_title).movie_id
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        orm.create_or_update_movie_user(movie_id, user_id, None)

        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Попередній фільм більше не буде пропонуватись в опитуванні.\nОцініть цей фільм:",
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
    elif message.text == "/пропустити":
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Попередній фільм було пропущено.\nОцініть цей фільм:",
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
                "Оцініть цей фільм:",
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
                "Рейтинг фільму повинен бути в межах [0, 10]\nОцініть цей фільм:",
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
