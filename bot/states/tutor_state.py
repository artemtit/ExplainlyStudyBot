from aiogram.fsm.state import State, StatesGroup


class TutorState(StatesGroup):
    awaiting_problem = State()
    awaiting_mode = State()
    learning_in_progress = State()
    solution_in_progress = State()
