"""Generic actor implementation, using torch."""

from robolfd import core
from robolfd import types
from robolfd.utils import pytorch_util as ptu

import torch
from torch import nn

import numpy as np
from typing import cast

class FeedForwardActor(core.Actor):
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions.
  """

  def __init__(
      self,
      policy_network: nn.Module,
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
    """
    # Store these for later use.
    self._policy_network = policy_network

  def select_action(self, obs: types.NestedArray) -> types.NestedArray:
    # Pass the observation through the policy network
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation_tensor = torch.tensor(observation, dtype=torch.float).to(ptu.device)
        if not self._policy_network.deterministic:
            action_distribution = self._policy_network.forward(observation_tensor)
            action = cast(
                np.ndarray,
                action_distribution.sample().cpu().detach().numpy(),
            )[0]
        else:
            action = self._policy_network.forward(observation_tensor)
            action = ptu.to_numpy(action)[0]
        return action
