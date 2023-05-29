import telebot


def home_markup():
    home_keyboard = telebot.types.ReplyKeyboardMarkup()
    home_keyboard.row("/recommend_film", "/test")
    return home_keyboard


def cancel_markup():
    cancel_keyboard = telebot.types.ReplyKeyboardMarkup()
    cancel_keyboard.row("/to_main_menu")
    return cancel_keyboard


def confirm_test_markup():
    cancel_keyboard = telebot.types.ReplyKeyboardMarkup()
    cancel_keyboard.row("/start_test")
    return cancel_keyboard


def rating_markup():
    rating_keyboard = telebot.types.ReplyKeyboardMarkup()
    rating_keyboard.row("/skip", "1", "2", "3", "4", "5", "/end")
    return rating_keyboard
