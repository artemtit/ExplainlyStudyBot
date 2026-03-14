from aiogram.fsm.state import State, StatesGroup


class StudyState(StatesGroup):
    awaiting_topic = State()
    awaiting_explanation_level = State()
    in_lesson = State()
    in_flashcards = State()
    in_test = State()
    in_practice = State()
