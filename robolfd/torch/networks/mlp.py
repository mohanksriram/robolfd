from robolfd.utils import pytorch_util as ptu

import numpy as np
import torch
from torch import nn
from torch import distributions
from typing import cast

class MLP(nn.Module):

    def __init__(self,
                in_dim,
                out_dim,
                n_layers,
                size,
                discrete=False,
                training=True,
                deterministic=True,
                **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.size = size
        self.deterministic = deterministic
        self.discrete = discrete
        self.training = training
        
        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.in_dim,
                output_size=self.out_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.in_dim,
                output_size=self.out_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.mean_net.to(ptu.device)
            if not self.deterministic:
                self.logstd = nn.Parameter(
                    torch.zeros(self.out_dim, dtype=torch.float32, device=ptu.device)
                )
                self.logstd.to(ptu.device)

    def forward(self, observation: torch.Tensor) -> distributions.Distribution:
        if self.discrete:
            return distributions.Categorical(logits=self.logits_na(observation))
        else:
            if not self.deterministic:
                return distributions.Normal(
                    self.mean_net(observation),
                    torch.exp(self.logstd)[None],
                )
            else:
                return self.mean_net(observation)
