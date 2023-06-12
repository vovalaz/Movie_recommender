import os
import telebot
import torch
import random

from tg_bot.bot import bot
from tg_bot import markups
from tg_bot.states import TestStates
from db import orm
from rec_system import content_collab_filtering as ccf

user_r = random.randint(1, 9999)
TMDB_KEY = os.getenv("TMDB_KEY")


@bot.message_handler(commands=["почати", "допомога"])
async def start_message(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
    orm.create_user_if_not_exists(message.from_user.username)
    if orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
        await bot.reply_to(
            message, "Привіт, я бот для рекомендацій в кінематографі.\nОберіть дію", reply_markup=markups.rec_markup()
        )
    else:
        await bot.reply_to(
            message, "Привіт, я бот для рекомендацій в кінематографі.\nОберіть дію", reply_markup=markups.home_markup()
        )


@bot.message_handler(state=TestStates.home_page, commands=["меню"])
async def menu_message(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
    if orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
        await bot.reply_to(
            message, "Привіт, я бот для рекомендацій в кінематографі.\nОберіть дію", reply_markup=markups.rec_markup()
        )
    else:
        await bot.reply_to(
            message, "Привіт, я бот для рекомендацій в кінематографі.\nОберіть дію", reply_markup=markups.home_markup()
        )


@bot.message_handler(state=TestStates.home_page, commands=["рекомендація"])
async def film_recommendation_handler(message: telebot.types.Message):
    model = ccf.RecommendationSystem(ccf.num_users, ccf.num_movies, ccf.embedding_dim)
    model.load_state_dict(torch.load("content_collab_model.pt"))
    user_id = orm.get_user_by_username(message.from_user.username).user_id
    rated_movies = orm.list_movies_rated_by_user(user_id)
    ratings = []
    for movie in rated_movies:
        rating = orm.get_movie_user(movie.movie_id, user_id).rating
        ratings.append((user_id + user_r, movie.movie_id, rating))

    recommended_film_title = ccf.generate_movie_recommendations(model, user_id + user_r, ratings, 1)[0]
    await bot.set_state(message.from_user.id, TestStates.recommend_film, message.chat.id)
    await bot.reply_to(
        message,
        "Рекомендація для вас:",
    )
    sent_title = await bot.send_message(
        message.chat.id,
        recommended_film_title,
        reply_markup=markups.rating_recommendation_markup(),
    )
    await bot.send_message(message.chat.id, orm.get_movie_link_by_title(recommended_film_title))
    orm.create_message(
        orm.get_user_by_username(message.from_user.username).user_id,
        sent_title.text,
        sent_title.chat.id,
        True,
    )


@bot.message_handler(state=TestStates.recommend_film)
async def recommendation_handler(message: telebot.types.Message):
    if message.text == "/закінчити":
        await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
        await bot.reply_to(message, "Рейтинги збережено.\nОберіть дію", reply_markup=markups.rec_markup())

    elif message.text == "/не_рекомендувати":
        movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
        movie_id = orm.get_movie_by_title(movie_title).movie_id
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        orm.create_or_update_movie_user(movie_id, user_id, None)

        model = ccf.RecommendationSystem(ccf.num_users, ccf.num_movies, ccf.embedding_dim)
        model.load_state_dict(torch.load("content_collab_model.pt"))
        user_id = orm.get_user_by_username(message.from_user.username).user_id
        rated_movies = orm.list_movies_rated_by_user(user_id)
        ratings = []
        for movie in rated_movies:
            rating = orm.get_movie_user(movie.movie_id, user_id).rating
            ratings.append((user_id + user_r, movie.movie_id, rating))

        recommended_film_title = ccf.generate_movie_recommendations(model, user_id + user_r, ratings, 1)[0]
        while recommended_film_title in [movie.title for movie in orm.list_movies_rated_by_user(user_id)]:
            recommended_film_title = ccf.generate_movie_recommendations(model, user_id + user_r, ratings, 1)[0]
        await bot.reply_to(
            message,
            "Рекомендація для вас:",
        )
        sent_title = await bot.send_message(
            message.chat.id,
            recommended_film_title,
            reply_markup=markups.rating_recommendation_markup(),
        )
        await bot.send_message(message.chat.id, orm.get_movie_link_by_title(recommended_film_title))
        orm.create_message(
            orm.get_user_by_username(message.from_user.username).user_id,
            sent_title.text,
            sent_title.chat.id,
            True,
        )
    elif message.text.isnumeric():
        rating = int(message.text)
        if 0 <= rating <= 10:
            movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
            movie_id = orm.get_movie_by_title(movie_title).movie_id
            user_id = orm.get_user_by_username(message.from_user.username).user_id
            orm.create_or_update_movie_user(movie_id, user_id, rating)

            model = ccf.RecommendationSystem(ccf.num_users, ccf.num_movies, ccf.embedding_dim)
            model.load_state_dict(torch.load("content_collab_model.pt"))
            user_id = orm.get_user_by_username(message.from_user.username).user_id
            rated_movies = orm.list_movies_rated_by_user(user_id)
            ratings = []
            for movie in rated_movies:
                rating = orm.get_movie_user(movie.movie_id, user_id).rating
                ratings.append((user_id + user_r, movie.movie_id, rating))

            recommended_film_title = ccf.generate_movie_recommendations(model, user_id + user_r, ratings, 1)[0]
            while recommended_film_title in [movie.title for movie in orm.list_movies_rated_by_user(user_id)]:
                recommended_film_title = ccf.generate_movie_recommendations(model, user_id + user_r, ratings, 1)[0]
            await bot.reply_to(
                message,
                "Рекомендація для вас:",
            )
            sent_title = await bot.send_message(
                message.chat.id,
                recommended_film_title,
                reply_markup=markups.rating_recommendation_markup(),
            )
            await bot.send_message(message.chat.id, orm.get_movie_link_by_title(recommended_film_title))
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_title.text,
                sent_title.chat.id,
                True,
            )
        else:
            await bot.send_message(
                message.chat.id,
                "Рейтинг фільму повинен бути в межах [0, 10]\nОцініть цей фільм знову:",
            )
            movie_title = orm.get_last_message_by_chat_id(message.chat.id).message_text
            sent_movie_title = await bot.send_message(
                message.chat.id,
                str(movie_title),
                reply_markup=markups.rating_recommendation_markup(),
            )
            await bot.send_message(message.chat.id, orm.get_movie_link_by_title(movie_title))
            orm.create_message(
                orm.get_user_by_username(message.from_user.username).user_id,
                sent_movie_title.text,
                sent_movie_title.chat.id,
                True,
            )


@bot.message_handler(state=TestStates.home_page, commands=["опитування"])
async def test_start_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.test, message.chat.id)
    await bot.send_message(
        message.chat.id,
        "Вам буде запропоновано оцінити фільми (від 1 до 10) для того щоб створити персоналізовані рекомендації.\nВи можете завершити оцінювання в будь-який момент. Всі рейтинги будуть збережені",
        reply_markup=markups.confirm_test_markup(),
    )


@bot.message_handler(state="*", commands=["на_головну"])
async def cancel_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
    orm.create_user_if_not_exists(message.from_user.username)
    if orm.list_movies_rated_by_user(orm.get_user_by_username(message.from_user.username).user_id):
        await bot.reply_to(message, "Оберіть іншу дію", reply_markup=markups.rec_markup())
    else:
        await bot.reply_to(message, "Оберіть іншу дію", reply_markup=markups.home_markup())
