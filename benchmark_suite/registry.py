from __future__ import annotations

from .task_defs.gec import build_task as build_gec_task
from .task_defs.intent_classification import build_task as build_intent_task
from .task_defs.legal_classification import build_task as build_legal_task
from .task_defs.machine_translation import build_task as build_mt_task
from .task_defs.named_entity_recognition import build_task as build_ner_task
from .task_defs.pos_tagging import build_task as build_pos_task
from .task_defs.summarization import build_task as build_summarization_task
from .spec import TaskSpec


TASKS: dict[str, TaskSpec] = {
    "gec": build_gec_task(),
    "intent_classification": build_intent_task(),
    "legal_classification": build_legal_task(),
    "machine_translation": build_mt_task(),
    "ner": build_ner_task(),
    "pos": build_pos_task(),
    "summarization": build_summarization_task(),
}


def list_tasks() -> list[str]:
    return sorted(TASKS)
