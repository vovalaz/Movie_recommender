from telebot.asyncio_handler_backends import State, StatesGroup


class TestStates(StatesGroup):
    home_page = State()
    recommend_film = State()
    test = State()
    q1 = State()
