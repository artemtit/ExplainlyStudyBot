from aiogram.fsm.state import State, StatesGroup


class StudyState(StatesGroup):
    awaiting_topic = State()
    material_ready = State()
    passing_test = State()
