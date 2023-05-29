import telebot

from tg_bot.bot import bot
from tg_bot import markups
from tg_bot.states import TestStates
from db import orm


@bot.message_handler(commands=["start", "help"])
async def start_message(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
    await bot.reply_to(
        message, "Hi!\nI'm movie recommendation bot.\nWhat you want to do?", reply_markup=markups.home_markup()
    )


@bot.message_handler(state=TestStates.home_page, commands=["menu"])
async def menu_message(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
    await bot.reply_to(
        message, "Hi!\nI'm movie recommendation bot.\nWhat you want to do?", reply_markup=markups.home_markup()
    )


@bot.message_handler(state=TestStates.home_page, commands=["recommend_film"])
async def film_recommendation_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.recommend_film, message.chat.id)
    await bot.reply_to(message, "Film recommended for you", reply_markup=markups.home_markup())


@bot.message_handler(state=TestStates.home_page, commands=["test"])
async def test_start_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.test, message.chat.id)
    await bot.send_message(
        message.chat.id,
        "You will be prompted to rate the films to create your personalized recommendations",
        reply_markup=markups.confirm_test_markup(),
    )
    orm.create_user_if_not_exists(message.from_user.username)


@bot.message_handler(state="*", commands=["to_main_menu"])
async def cancel_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, TestStates.home_page, message.chat.id)
    await bot.send_message(message.chat.id, "Choose other option", reply_markup=markups.home_markup())
