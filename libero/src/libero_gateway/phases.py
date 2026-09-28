from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class PhaseCondition:
    """One server-side condition used to mark a task phase as achieved."""

    kind: str
    arguments: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PhaseSpec:
    """Stable public metadata and private evaluator input for one task phase."""

    phase_id: int
    name: str
    description: str
    condition: PhaseCondition

    def public_definition(self) -> Dict[str, object]:
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "description": self.description,
        }


def _phase(
    phase_id: int,
    name: str,
    description: str,
    kind: str,
    *arguments: str,
) -> PhaseSpec:
    return PhaseSpec(
        phase_id=phase_id,
        name=name,
        description=description,
        condition=PhaseCondition(kind, arguments),
    )


def _two_object_task(
    first_name: str,
    first_description: str,
    first_goal: Tuple[str, ...],
    second_name: str,
    second_description: str,
) -> Tuple[PhaseSpec, ...]:
    return (
        _phase(
            1,
            f"grasp_{first_name}",
            f"Grasp {first_description}.",
            "grasped",
            first_name,
        ),
        _phase(
            2,
            f"place_{first_name}",
            f"Place {first_description} at its goal.",
            "predicate",
            *first_goal,
        ),
        _phase(
            3,
            f"grasp_{second_name}",
            f"Grasp {second_description}.",
            "grasped",
            second_name,
        ),
        _phase(
            4,
            "complete_task",
            "Satisfy the complete LIBERO task goal.",
            "success",
        ),
    )


LIBERO10_PHASES: Dict[int, Tuple[PhaseSpec, ...]] = {
    0: _two_object_task(
        "alphabet_soup_1",
        "the alphabet soup",
        ("in", "alphabet_soup_1", "basket_1_contain_region"),
        "tomato_sauce_1",
        "the tomato sauce",
    ),
    1: _two_object_task(
        "cream_cheese_1",
        "the cream cheese box",
        ("in", "cream_cheese_1", "basket_1_contain_region"),
        "butter_1",
        "the butter",
    ),
    2: (
        _phase(
            1,
            "turn_on_stove",
            "Turn on the stove.",
            "predicate",
            "turnon",
            "flat_stove_1",
        ),
        _phase(2, "grasp_moka_pot", "Grasp the moka pot.", "grasped", "moka_pot_1"),
        _phase(3, "lift_moka_pot", "Lift the moka pot at least 2 cm.", "lifted", "moka_pot_1"),
        _phase(4, "complete_task", "Place the moka pot on the switched-on stove.", "success"),
    ),
    3: (
        _phase(1, "grasp_black_bowl", "Grasp the black bowl.", "grasped", "akita_black_bowl_1"),
        _phase(
            2,
            "lift_black_bowl",
            "Lift the black bowl at least 2 cm.",
            "lifted",
            "akita_black_bowl_1",
        ),
        _phase(
            3,
            "put_bowl_in_bottom_drawer",
            "Put the black bowl in the bottom drawer.",
            "predicate",
            "in",
            "akita_black_bowl_1",
            "white_cabinet_1_bottom_region",
        ),
        _phase(4, "complete_task", "Close the bottom drawer with the bowl inside.", "success"),
    ),
    4: _two_object_task(
        "porcelain_mug_1",
        "the white mug",
        ("on", "porcelain_mug_1", "plate_1"),
        "white_yellow_mug_1",
        "the yellow-and-white mug",
    ),
    5: (
        _phase(
            1,
            "reach_book",
            "Move the end effector within 8 cm of the book.",
            "near",
            "black_book_1",
        ),
        _phase(2, "grasp_book", "Grasp the book.", "grasped", "black_book_1"),
        _phase(3, "lift_book", "Lift the book at least 2 cm.", "lifted", "black_book_1"),
        _phase(4, "complete_task", "Place the book in the caddy's back compartment.", "success"),
    ),
    6: _two_object_task(
        "porcelain_mug_1",
        "the white mug",
        ("on", "porcelain_mug_1", "plate_1"),
        "chocolate_pudding_1",
        "the chocolate pudding",
    ),
    7: _two_object_task(
        "alphabet_soup_1",
        "the alphabet soup",
        ("in", "alphabet_soup_1", "basket_1_contain_region"),
        "cream_cheese_1",
        "the cream cheese box",
    ),
    8: _two_object_task(
        "moka_pot_1",
        "the first moka pot",
        ("on", "moka_pot_1", "flat_stove_1_cook_region"),
        "moka_pot_2",
        "the second moka pot",
    ),
    9: (
        _phase(1, "grasp_mug", "Grasp the yellow-and-white mug.", "grasped", "white_yellow_mug_1"),
        _phase(
            2,
            "put_mug_in_microwave",
            "Put the mug in the microwave heating region.",
            "predicate",
            "in",
            "white_yellow_mug_1",
            "microwave_1_heating_region",
        ),
        _phase(
            3,
            "close_microwave",
            "Close the microwave after inserting the mug.",
            "predicate",
            "close",
            "microwave_1",
        ),
        _phase(4, "complete_task", "Leave the mug inside the closed microwave.", "success"),
    ),
}


def phases_for_task(benchmark: str, task_id: int) -> Tuple[PhaseSpec, ...]:
    """Return phase definitions for a task, or no phases outside LIBERO-10."""

    if benchmark != "libero_10":
        return ()
    return LIBERO10_PHASES[task_id]


class PhaseTracker:
    """Track independent, monotonic phase achievements within one Episode."""

    def __init__(self, phases: Tuple[PhaseSpec, ...]):
        self.phases = phases
        self._first_achieved_steps: Dict[int, Optional[int]] = {}
        self.reset()

    def reset(self) -> None:
        self._first_achieved_steps = {phase.phase_id: None for phase in self.phases}

    def update(
        self,
        step: int,
        evaluate: Callable[[PhaseSpec], bool],
    ) -> None:
        """Record every phase condition satisfied at this step."""

        for phase in self.phases:
            if self._first_achieved_steps[phase.phase_id] is not None:
                continue
            if evaluate(phase):
                self._first_achieved_steps[phase.phase_id] = step

    def statuses(self) -> Tuple[Dict[str, object], ...]:
        return tuple(
            {
                **phase.public_definition(),
                "success": self._first_achieved_steps[phase.phase_id] is not None,
                "first_achieved_step": self._first_achieved_steps[phase.phase_id],
            }
            for phase in self.phases
        )
