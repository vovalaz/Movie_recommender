import asyncio

from tg_bot import message_handlers
from tg_bot import testing_handlers

from telebot import asyncio_filters

from tg_bot.bot import bot
from init_db import init_db


if __name__ == "__main__":
    # init_db()
    bot.add_custom_filter(asyncio_filters.StateFilter(bot))
    asyncio.run(bot.polling())
    bot.close()
