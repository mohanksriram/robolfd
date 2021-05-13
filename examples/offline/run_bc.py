import robolfd
from robolfd import types
from robolfd.agents.torch.bc import learning
from robolfd.utils import pytorch_util as ptu
from robolfd.utils import counting
from robolfd.utils import loggers

import torch
from torch import nn
from torch import distributions

results_dir = "/tmp/robosuite/bc"
learning_rate = 2e-4
batch_size = 16

#TODO: Move policy net to a separate neural network module
class PolicyNet(nn.Module):

    def __init__(self,
                ac_dim,
                ob_dim,
                n_layers,
                size,
                discrete=False,
                training=True,
                nn_baseline=False,
                **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)

    def forward(self, observation: torch.Tensor) -> distributions.Distribution:
        if self.discrete:
            return distributions.Categorical(logits=self.logits_na(observation))
        else:
            return distributions.Normal(
                self.mean_net(observation),
                torch.exp(self.logstd)[None],
            )

# get_action method should be move to a separate actor




