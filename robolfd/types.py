"""Common types used throughout robolfd."""

import enum
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Optional, Union
from dataclasses import dataclass


NestedArray = Any
NestedTensor = Any

class Transition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()

@dataclass
class Trajectory:
    """Represents a sequence of transitions"""

    observations: Optional[NestedArray]
    actions: Optional[NestedArray]
    next_observations: Optional[NestedArray]
    rewards: Optional[NestedArray]
    dones: Optional[NestedArray]

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index: int) -> Transition:
        return Transition(
            self.observations[index],
            self.actions[index],
            self.next_observations[index],
            self.rewards[index],
            self.dones[index],
        )

class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""
    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST

class TimeStep(NamedTuple):
  """Returned with every call to `step` and `reset` on an environment.
  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
  NumPy array or a dict or list of arrays), and an associated `reward` and
  `discount`.
  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.
  Attributes:
    step_type: A `StepType` enum value.
    reward:  A scalar, NumPy array, nested dict, list or tuple of rewards; or
      `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
      sequence.
    discount: A scalar, NumPy array, nested dict, list or tuple of discount
      values in the range `[0, 1]`, or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    observation: A NumPy array, or a nested dict, list or tuple of arrays.
      Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
      also valid in place of a scalar array.
  """

# TODO(b/143116886): Use generics here when PyType supports them.
step_type: Any
reward: Any
discount: Any
observation: Any

def first(self) -> bool:
    return self.step_type == StepType.FIRST

def mid(self) -> bool:
    return self.step_type == StepType.MID

def last(self) -> bool:
    return self.step_type == StepType.LAST
