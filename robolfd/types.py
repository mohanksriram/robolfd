"""Common types used throughout robolfd."""

from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union

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
