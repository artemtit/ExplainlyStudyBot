from __future__ import annotations

from ai.llm_client import LlmClient
from database.repository import Repository
from services.analytics_service import AnalyticsService
from services.history_service import HistoryService
from services.lesson_service import LessonService
from services.question_service import QuestionService
from services.session_service import SessionService
from services.streak_service import StreakService
from services.topic_suggestion_service import TopicSuggestionService

llm_client = LlmClient()
repository = Repository()
analytics_service = AnalyticsService()
session_service = SessionService()
streak_service = StreakService()
lesson_service = LessonService(
    llm_client,
    repository=repository,
    analytics=analytics_service,
    session_service=session_service,
    streak_service=streak_service,
)
question_service = QuestionService(llm_client)
history_service = HistoryService(repository)
topic_suggestion_service = TopicSuggestionService()
