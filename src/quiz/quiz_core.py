"""Core quiz logic: questions, answer handling, and session/result flow.

This module has no dependency on the recommendation model internals —
per the team guide, the quiz UI/logic must stay decoupled from model
code. It only knows how to run a quiz and produce plain Python data
that `interaction_events.py` turns into pipeline-ready rows.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


class InvalidAnswerError(ValueError):
    """Raised when a submitted answer isn't one of the question's options."""


@dataclass
class Question:
    """A single quiz question."""

    question_id: int
    category: str
    difficulty: str  # "easy" | "medium" | "hard"
    text: str
    options: list[str]
    correct_answer: str

    def __post_init__(self) -> None:
        if self.correct_answer not in self.options:
            raise ValueError(
                f"correct_answer {self.correct_answer!r} must be one of "
                f"options {self.options!r} for question {self.question_id}"
            )
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(
                f"difficulty must be one of easy/medium/hard, got "
                f"{self.difficulty!r}"
            )

    def is_correct(self, selected_answer: str) -> bool:
        return selected_answer == self.correct_answer


@dataclass
class AnsweredQuestion:
    """The result of a user answering one question."""

    user_id: int
    quiz_session_id: str
    question: Question
    selected_answer: str
    attempt_number: int
    timestamp: float
    is_correct: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_correct = self.question.is_correct(self.selected_answer)


class QuizSession:
    """Runs a single quiz attempt for one user across a set of questions.

    Usage:
        session = QuizSession(user_id=101, questions=[q1, q2, q3])
        session.answer(question_id=q1.question_id, selected_answer="B")
        session.answer(question_id=q2.question_id, selected_answer="A")
        results = session.finish()
    """

    def __init__(
        self,
        user_id: int,
        questions: list[Question],
        attempt_counts: dict[int, int] | None = None,
        session_id: str | None = None,
    ) -> None:
        if not questions:
            raise ValueError("QuizSession requires at least one question")

        self.user_id = user_id
        self.questions = {q.question_id: q for q in questions}
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        # tracks how many times this user has answered each question_id,
        # across sessions, so `attempt_number` is meaningful over time.
        self._attempt_counts: dict[int, int] = dict(attempt_counts or {})
        self.answered: list[AnsweredQuestion] = []
        self._finished = False

    def answer(self, question_id: int, selected_answer: str) -> AnsweredQuestion:
        if self._finished:
            raise RuntimeError("Cannot answer questions after finish() was called")
        if question_id not in self.questions:
            raise KeyError(f"question_id {question_id} is not part of this session")

        question = self.questions[question_id]
        if selected_answer not in question.options:
            raise InvalidAnswerError(
                f"{selected_answer!r} is not a valid option for question "
                f"{question_id}. Valid options: {question.options}"
            )

        self._attempt_counts[question_id] = self._attempt_counts.get(question_id, 0) + 1

        result = AnsweredQuestion(
            user_id=self.user_id,
            quiz_session_id=self.session_id,
            question=question,
            selected_answer=selected_answer,
            attempt_number=self._attempt_counts[question_id],
            timestamp=time.time(),
        )
        self.answered.append(result)
        return result

    def score(self) -> float:
        """Fraction of answered questions that were correct (0.0-1.0)."""
        if not self.answered:
            return 0.0
        correct = sum(1 for a in self.answered if a.is_correct)
        return correct / len(self.answered)

    def finish(self) -> list[AnsweredQuestion]:
        """Mark the session complete and return all answered questions."""
        self._finished = True
        return self.answered
