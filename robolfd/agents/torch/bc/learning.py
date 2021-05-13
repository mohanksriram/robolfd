from typing import List

import robolfd
from robolfd.utils import counting
from robolfd.utils import loggers
import numpy as np

import torch
from torch import nn


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

        # Get an iterator over the dataset.
        self._iterator = iter(dataset)

        #TODO
        self._optimizer = None
        self._network = network
        
        #TODO
        self._variables = None
        self._num_steps = 0
        
        #TODO
        # Create a snapshotter object.
        self._snapshotter = None

    def step(self):
        # TODO Do a batch of SGD.
        result = None

        # Update our counts and record it.
        counts = self._counter.increment(steps=1)
        result.update(counts)

        # Attempt to write logs.
        self._logger.write(result)

    def get_variables(self, names: List[str]) -> List[np.ndarray]:
        #TODO
        return None
        #return tf2_utils.to_numpy(self._variables)
    
    def save(self, filepath):
        torch.save(self._network.state_dict(), filepath)

    @property
    def state(self):
        """Returns the stateful parts of the learner for checkpointing."""
        return {
            'network': self._network,
            'optimizer': self._optimizer,
            'num_steps': self._num_steps
        }