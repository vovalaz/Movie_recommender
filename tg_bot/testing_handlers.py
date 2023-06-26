import random
import telebot

from db import orm
from tg_bot.bot import bot
from tg_bot import markups
from tg_bot.states import TestStates


# Message handler for test management
@bot.message_handler(state=TestStates.test)
async def test_handler(message: telebot.types.Message):
    # start of test option
    if message.text == "/почати_опитування":
        # process of finding movie that is not watched by user
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Оцініть цей фільм:",
        )
        # sending movie title to user
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        # sending movie link to user
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    # end test option
    elif message.text == "/закінчити":
        # changing state to home page
        await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
        await bot.reply_to(message, "Рейтинги збережено.\nОберіть наступну дію", reply_markup=markups.rec_markup())
    # don't recommend (add to blacklist) option
    elif message.text == "/не_рекомендувати":
        movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
        movie_id = orm.get_movie_by_title(movie_title).movie_id
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        # adding to blacklist
        orm.create_or_update_movie_user(movie_id, user_id, None)

        # process of finding next movie
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Попередній фільм більше не буде пропонуватись в опитуванні.\nОцініть цей фільм:",
        )
        # sending movie title to user
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        # sending movie link to user
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    # skip movie option
    elif message.text == "/пропустити":
        # process of finding next movie
        movie = orm.get_movie(random.randint(1, 500))
        while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
            movie = orm.get_movie(random.randint(1, 500))

        await bot.send_message(
            message.chat.id,
            "Попередній фільм було пропущено.\nОцініть цей фільм:",
        )
        # sending movie title to user
        sent_movie_title = await bot.send_message(
            message.chat.id,
            str(movie.title),
            reply_markup=markups.rating_markup(),
        )
        # sending movie link to user
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    # rating option
    # checking if message user sent is numeric
    elif message.text.isnumeric():
        rating = int(message.text)
        # validating message
        if 0 <= rating <= 10:
            movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
            movie_id = orm.get_movie_by_title(movie_title).movie_id
            user_id = orm.get_user_by_username(message.from_user.username).user_id
            # create rating for movie
            orm.create_or_update_movie_user(movie_id, user_id, int(message.text))

            # find other movie
            movie = orm.get_movie(random.randint(1, 500))
            while movie in orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
                movie = orm.get_movie(random.randint(1, 500))

            await bot.send_message(
                message.chat.id,
                "Оцініть цей фільм:",
            )
            # sending movie title to user
            sent_movie_title = await bot.send_message(
                message.chat.id,
                str(movie.title),
                reply_markup=markups.rating_markup(),
            )
            # sending movie link to user
            await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie.title)))
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_movie_title.text,
                sent_movie_title.chat.id,
                True,
            )
        # if message is not numeric
        else:
            # send error message and ask to rate again
            await bot.send_message(
                message.chat.id,
                "Рейтинг фільму повинен бути в межах [0, 10]\nОцініть цей фільм:",
            )
            # getting last sent movie title and send it
            movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
            sent_movie_title = await bot.send_message(
                message.chat.id,
                str(movie_title),
                reply_markup=markups.rating_markup(),
            )
            # sending movie link to user
            await bot.send_message(message.chat.id, orm.get_movie_link_by_title(str(movie_title)))
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_movie_title.text,
                sent_movie_title.chat.id,
                True,
            )
