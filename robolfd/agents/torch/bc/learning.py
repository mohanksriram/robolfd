import itertools

from numpy.random import sample
from robolfd.types import Transition
from typing import List

import robolfd
from robolfd.utils import counting
from robolfd.utils import loggers
from robolfd.utils import pytorch_util as ptu
import numpy as np

import torch
from torch import nn
from torch import optim


class BCLearner(robolfd.Learner):
    """BC learner.

    This is the learning component of a BC agent. ie. it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(self,
                 network: nn.Module,
                 learning_rate,
                 dataset,
                 counter,
                 logger: loggers.Logger,
                 batch_size = 32,
                 checkpoint: bool = True):
        """Intializes the learner.

         Args:
            network: the BC network (the one being optimized)
            learning_rate: learning rate for the cross-entropy update.
            dataset: dataset to learn from.
            counter: Counter object for (potentially distributed) counting.
            logger: Logger object for writing logs to.
            checkpoint: boolean indicating whether to checkpoint the learner.
        """

        self._counter = counter or counting.Counter()
        self._logger = logger

        # Get an iterator over the dataset.
        self._dataset = dataset
        self._learning_rate = learning_rate
        self._batch_size = batch_size

        #TODO
        self._network = network
        self._discrete = network.discrete
        
        if self._discrete:
            self._optimizer = optim.Adam(self._network.logits_na.parameters(),
                                            self._learning_rate)
        else:
            self._optimizer = optim.Adam(
                itertools.chain([self._network.logstd], self._network.mean_net.parameters()),
                self._learning_rate
            )
        
        #TODO
        self._variables = None
        self._num_steps = 0
        
        #TODO
        # Create a snapshotter object.
        self._snapshotter = None

    def _step(self):
        self._optimizer.zero_grad()
        transitions: Transition = self.sample_bs(self._dataset, self._batch_size)
        observations = transitions[0]
        torch_observations = torch.tensor(observations, device='cpu', dtype=torch.float)

        torch_actions = torch.tensor(
            transitions[1], device='cpu',
            dtype=torch.int if self._discrete else torch.float)

        action_distribution = self._network(torch_observations)
        
        # Loss is proportional to the negative log-likelihood.
        loss = -action_distribution.log_prob(torch_actions).mean()
        loss.backward()
        self._optimizer.step()

        return {
            'Training Loss': ptu.to_numpy(loss)
        }


    def step(self):
        # TODO Do a batch of SGD.
        result = self._step()
        
        # Update our counts and record it.
        counts = self._counter.increment(steps=1)
        result.update(counts)

        # Attempt to write logs.
        self._logger.write(result)
    
    def save(self, filepath):
        torch.save(self._network.state_dict(), filepath)

    def sample_bs(self, dataset, batch_size: int):
        sample_indices = np.random.permutation(len(dataset))[:batch_size]
        return dataset[sample_indices]

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        #TODO
        return None
    #return tf2_utils.to_numpy(self._variables)

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'network': self._network,
            'optimizer': self._optimizer,
            'num_steps': self._num_steps
        }