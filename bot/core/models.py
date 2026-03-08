from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from bot.core.errors import MaterialValidationError

logger = logging.getLogger(__name__)

TEST_LETTERS = ("A", "B", "C", "D")
MAX_TESTS = 5


def _log_validation_warning(message: str, **context: Any) -> None:
    extra = {key: value for key, value in context.items() if value is not None}
    if extra:
        logger.warning(message, extra=extra)
    else:
        logger.warning(message)


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    question: str
    options: list[str]
    correct: str
    explanation: str

    @classmethod
    def from_raw(
        cls,
        raw: Mapping[str, Any],
        *,
        topic: str | None = None,
        index: int | None = None,
    ) -> "QuizQuestion":
        question = str(raw.get("question") or "").strip()
        if not question:
            raise MaterialValidationError("Test question is missing")

        options = cls._coerce_options(raw.get("options"))
        if len(options) != 4 or any(not option for option in options):
            raise MaterialValidationError("Malformed test options")

        correct_raw = str(raw.get("correct") or "").strip().upper()
        if correct_raw not in TEST_LETTERS:
            answer_raw = str(raw.get("answer") or "").strip()
            if answer_raw:
                for idx, option in enumerate(options):
                    if answer_raw.lower() == option.lower():
                        correct_raw = TEST_LETTERS[idx]
                        break
        if correct_raw not in TEST_LETTERS:
            raise MaterialValidationError("Invalid test correct answer")

        explanation = str(raw.get("explanation") or "").strip()
        if not explanation:
            explanation = "\u2014"
            _log_validation_warning(
                "Test explanation missing, using fallback",
                topic=topic,
                test_index=index,
            )

        return cls(question=question, options=options, correct=correct_raw, explanation=explanation)

    @classmethod
    def list_from_raw(cls, raw: Any, *, topic: str | None = None) -> list["QuizQuestion"]:
        if not isinstance(raw, list):
            raise MaterialValidationError("Tests payload must be a list")
        if not raw:
            raise MaterialValidationError("Tests payload is empty")

        tests: list[QuizQuestion] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise MaterialValidationError(f"Test item at index {idx} must be an object")
            tests.append(cls.from_raw(item, topic=topic, index=idx))

        return tests[:MAX_TESTS]

    @staticmethod
    def _coerce_options(raw: Any) -> list[str]:
        if isinstance(raw, Mapping):
            options = [str(raw.get(letter, "")).strip() for letter in TEST_LETTERS]
        elif isinstance(raw, list):
            options = [str(option).strip() for option in raw]
        else:
            raise MaterialValidationError("Test options must be a list or object")

        if len(options) < 4:
            return options
        return options[:4]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options),
            "correct": self.correct,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class LessonSection:
    header: str
    text: str
    key_points: list[str]
    formula: str | None

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "LessonSection":
        header = str(raw.get("header") or "\u0420\u0430\u0437\u0434\u0435\u043b").strip()
        text = str(raw.get("text") or "").strip()
        if not text:
            raise MaterialValidationError("Lesson section text is missing")

        key_points_raw = raw.get("key_points")
        if key_points_raw is None:
            key_points: list[str] = []
        elif isinstance(key_points_raw, list):
            key_points = [str(point).strip() for point in key_points_raw if str(point).strip()]
        else:
            raise MaterialValidationError("Lesson key_points must be a list")

        formula = raw.get("formula")
        if formula is not None:
            formula = str(formula)

        return cls(header=header, text=text, key_points=key_points, formula=formula)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header,
            "text": self.text,
            "key_points": list(self.key_points),
            "formula": self.formula,
        }


@dataclass(frozen=True, slots=True)
class Lesson:
    title: str
    sections: list[LessonSection]

    @classmethod
    def from_raw(cls, raw: Any) -> "Lesson":
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                raise MaterialValidationError("Lesson text is empty")
            sections = [
                LessonSection(
                    header="\u041e\u0431\u0437\u043e\u0440",
                    text=text,
                    key_points=[],
                    formula=None,
                )
            ]
            return cls(title="\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a", sections=sections)

        if not isinstance(raw, Mapping):
            raise MaterialValidationError("Lesson must be an object or string")

        title = str(raw.get("title") or "\u041c\u0438\u043d\u0438-\u0443\u0440\u043e\u043a").strip()
        sections_raw = raw.get("sections")
        sections: list[LessonSection] = []

        if isinstance(sections_raw, list):
            for idx, section in enumerate(sections_raw):
                if not isinstance(section, Mapping):
                    raise MaterialValidationError(f"Lesson section at index {idx} must be an object")
                sections.append(LessonSection.from_raw(section))

        if not sections:
            fallback_text = str(raw.get("text") or raw.get("content") or "").strip()
            if not fallback_text:
                raise MaterialValidationError("Lesson sections are missing")
            sections = [
                LessonSection(
                    header="\u041e\u0431\u0437\u043e\u0440",
                    text=fallback_text,
                    key_points=[],
                    formula=None,
                )
            ]

        return cls(title=title, sections=sections)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "sections": [section.to_dict() for section in self.sections]}


@dataclass(frozen=True, slots=True)
class Flashcard:
    question: str
    answer: str

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "Flashcard":
        question = str(raw.get("question") or "").strip()
        answer = str(raw.get("answer") or "").strip()
        if not question or not answer:
            raise MaterialValidationError("Card question/answer is missing")
        return cls(question=question, answer=answer)

    @classmethod
    def list_from_raw(cls, raw: Any) -> list["Flashcard"]:
        if not isinstance(raw, list):
            raise MaterialValidationError("Cards payload must be a list")
        if not raw:
            raise MaterialValidationError("Cards payload is empty")

        cards: list[Flashcard] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise MaterialValidationError(f"Card item at index {idx} must be an object")
            cards.append(cls.from_raw(item))

        return cards[:5]

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer}


@dataclass(frozen=True, slots=True)
class PracticeProblem:
    problem: str
    solution: str

    @classmethod
    def from_raw(cls, raw: Any) -> "PracticeProblem":
        if not isinstance(raw, Mapping):
            raise MaterialValidationError("Practice must be an object")
        problem = str(raw.get("problem") or "").strip()
        solution = str(raw.get("solution") or "").strip()
        if not problem or not solution:
            raise MaterialValidationError("Practice problem/solution is missing")
        return cls(problem=problem, solution=solution)

    def to_dict(self) -> dict[str, Any]:
        return {"problem": self.problem, "solution": self.solution}


@dataclass(frozen=True, slots=True)
class Material:
    lesson: Lesson
    cards: list[Flashcard]
    tests: list[QuizQuestion]
    practice: PracticeProblem

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], *, topic: str | None = None) -> "Material":
        if not isinstance(payload, Mapping):
            raise MaterialValidationError("Material payload must be an object")

        normalized = dict(payload)
        lesson_raw = normalized.get("lesson")
        if isinstance(lesson_raw, Mapping):
            for key in ("cards", "tests", "practice"):
                if key in lesson_raw and key not in normalized:
                    normalized[key] = lesson_raw[key]

        lesson = Lesson.from_raw(normalized.get("lesson"))
        cards = Flashcard.list_from_raw(normalized.get("cards"))
        tests = QuizQuestion.list_from_raw(normalized.get("tests"), topic=topic)
        practice = PracticeProblem.from_raw(normalized.get("practice"))

        return cls(lesson=lesson, cards=cards, tests=tests, practice=practice)

    def with_tests(self, tests: list[QuizQuestion]) -> "Material":
        return Material(lesson=self.lesson, cards=self.cards, tests=tests, practice=self.practice)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson": self.lesson.to_dict(),
            "cards": [card.to_dict() for card in self.cards],
            "tests": [test.to_dict() for test in self.tests],
            "practice": self.practice.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LearningSession:
    user_id: int
    topic: str
    stage: str
    card_index: int = 0
    test_index: int = 0
    test_score: int = 0
    started_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "topic": self.topic,
            "stage": self.stage,
            "card_index": self.card_index,
            "test_index": self.test_index,
            "test_score": self.test_score,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class TopicProgress:
    user_id: int
    topic: str
    completed: bool = False
    last_stage: str | None = None
    card_index: int = 0
    test_index: int = 0
    test_score: int = 0
    last_active_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "topic": self.topic,
            "completed": self.completed,
            "last_stage": self.last_stage,
            "card_index": self.card_index,
            "test_index": self.test_index,
            "test_score": self.test_score,
            "last_active_date": self.last_active_date,
        }


# Backwards-compatible aliases
TestQuestion = QuizQuestion
Card = Flashcard
Practice = PracticeProblem
