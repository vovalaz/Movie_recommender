import random
import telebot
import torch

from db import orm
from rec_system import content_collab_filtering as ccf
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
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_movie_title.text,
            sent_movie_title.chat.id,
            True,
        )
    elif message.text == "/end":
        await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
        await bot.reply_to(message, "Ratings saved.\nWhat you want to do next?", reply_markup=markups.home_markup())

        user_id = orm.get_user_by_username(message.from_user.username).user_id
        rated_movies = orm.list_movies_rated_by_user(user_id)
        train_data = []
        for movie in rated_movies:
            rating = orm.get_movie_user(movie.movie_id, user_id).rating
            train_data.append((user_id + 10000, movie.movie_id, rating))
        model = ccf.RecommendationSystem(ccf.num_users, ccf.num_movies, ccf.embedding_dim)
        model.load_state_dict(torch.load("content_collab_model.pt"))
        ccf.train_model(model, train_data, 10, ccf.learning_rate)

    elif message.text == "/skip":
        movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
        movie_id = orm.get_movie_by_title(movie_title).movie_id
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        orm.create_or_update_movie_user(movie_id, user_id, None)

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
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_movie_title.text,
                sent_movie_title.chat.id,
                True,
            )
