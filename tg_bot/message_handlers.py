import telebot

from tg_bot.bot import bot
from tg_bot.markups import cancel_markup, home_markup
from tg_bot.states import UploadStates


@bot.message_handler(commands=["start", "help"])
async def start_message(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, UploadStates.home_page, message.chat.id)
    await bot.reply_to(message, "Hi!\nI'm movie recommendation bot.\nWhat you want to do?", reply_markup=home_markup())


@bot.message_handler(state=UploadStates.home_page, commands=["menu"])
async def menu_message(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, UploadStates.home_page, message.chat.id)
    await bot.reply_to(message, "Hi!\nI'm movie recommendation bot.\nWhat you want to do?", reply_markup=home_markup())


@bot.message_handler(state=UploadStates.home_page, commands=["recommend_film"])
async def drive_management_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, UploadStates.recommend_film, message.chat.id)
    await bot.reply_to(message, "Film recommended for you", reply_markup=home_markup())


@bot.message_handler(state=UploadStates.home_page, commands=["test"])
async def upload_files_handler(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, UploadStates.direct_upload, message.chat.id)
    await bot.send_message(message.chat.id, "Send files", reply_markup=cancel_markup())


@bot.message_handler(state="*", commands=["to_main_menu"])
async def cancel_pick(message: telebot.types.Message):
    await bot.set_state(message.from_user.id, UploadStates.home_page, message.chat.id)
    await bot.send_message(message.chat.id, "Choose other option", reply_markup=home_markup())
