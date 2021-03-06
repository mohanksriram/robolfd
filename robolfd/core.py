"""Core Robolfd interfaces.

This file specifies and documents the notions of `Learner`.
"""

import abc
import itertools
from typing import Generic, List, Optional, Sequence, TypeVar

from robolfd import types
from robolfd.utils import metrics

T = TypeVar('T')

@metrics.record_class_usage
class Actor(abc.ABC):
  """Interface for an agent that can act.

  This interface defines an API for an Actor to interact with an EnvironmentLoop
  (see robolfd.environment_loop), e.g. a simple RL loop where each step is of the
  form:

    # Make the first observation.
    timestep = env.reset()
    actor.observe_first(timestep.observation)

    # Take a step and observe.
    action = actor.select_action(timestep.observation)
    next_timestep = env.step(action)
    actor.observe(action, next_timestep)

    # Update the actor policy/parameters.
    actor.update()
  """

  @abc.abstractmethod
  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    """Samples from the policy and returns an action."""

class VariableSource(abc.ABC):
  """Abstract source of variables.
  Objects which implement this interface provide a source of variables, returned
  as a collection of (nested) numpy arrays. Generally this will be used to
  provide variables to some learned policy/etc.
  """

  @abc.abstractmethod
  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    """Return the named variables as a collection of (nested) numpy arrays.
    Args:
      names: args where each name is a string identifying a predefined subset of
        the variables.
    Returns:
      A list of (nested) numpy arrays `variables` such that `variables[i]`
      corresponds to the collection named by `names[i]`.
    """


@metrics.record_class_usage
class Worker(abc.ABC):
  """An interface for (potentially) distributed workers."""

  @abc.abstractmethod
  def run(self):
    """Runs the worker."""

class Saveable(abc.ABC, Generic[T]):
  """An interface for saveable objects."""

  @abc.abstractmethod
  def save(self) -> T:
    """Returns the state from the object to be saved."""

  @abc.abstractmethod
  def restore(self, state: T):
    """Given the state, restores the object."""


class Learner(VariableSource, Worker, Saveable):
  """Abstract learner object.
  This corresponds to an object which implements a learning loop. A single step
  of learning should be implemented via the `step` method and this step
  is generally interacted with via the `run` method which runs update
  continuously.
  All objects implementing this interface should also be able to take in an
  external dataset (see torch.datasets) and run updates using data from this
  dataset. This can be accomplished by explicitly running `learner.step()`
  inside a for/while loop or by using the `learner.run()` convenience function.
  Data will be read from this dataset asynchronously and this is primarily
  useful when the dataset is filled by an external process.
  """

  @abc.abstractmethod
  def step(self):
    """Perform an update step of the learner's parameters."""

  def run(self, num_steps: Optional[int] = None) -> None:
    """Run the update loop; typically an infinite loop which calls step."""

    iterator = range(num_steps) if num_steps is not None else itertools.count()

    for _ in iterator:
      self.step()

  def save(self, filepath):
    raise NotImplementedError('Method "save" is not implemented.')

  def restore(self, state):
    raise NotImplementedError('Method "restore" is not implemented.')