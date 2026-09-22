"""Quiz / user-interaction module.

Owned by Nav. Produces interaction data consumed by the rest of the
recommendation pipeline via `load_interactions(path) -> DataFrame`.

Public surface:
    - Question, QuizSession        (quiz_core)
    - record_interaction           (interaction_events)
    - append_interaction           (interaction_events)
    - INTERACTION_COLUMNS          (interaction_events)
    - validate_interaction_schema  (interaction_events)
    - generate_synthetic_dataset   (data_generator)
"""

from .quiz_core import Question, QuizSession
from .interaction_events import (
    INTERACTION_COLUMNS,
    CSV_READ_DTYPES,
    record_interaction,
    append_interaction,
    append_interactions,
    read_interactions,
    validate_interaction_schema,
)
from .data_generator import generate_synthetic_dataset

__all__ = [
    "Question",
    "QuizSession",
    "INTERACTION_COLUMNS",
    "CSV_READ_DTYPES",
    "record_interaction",
    "append_interaction",
    "append_interactions",
    "read_interactions",
    "validate_interaction_schema",
    "generate_synthetic_dataset",
]
