from robolfd.agents.torch.bc import learning
import robolfd
from robolfd import types
from robolfd.agents.torch import bc
# from robolfd.agents.torch import actors
from robolfd.utils import pytorch_util as ptu
from robolfd.utils import counting
from robolfd.utils import loggers
from examples.offline import bc_robo_utils

from absl import app
from absl import flags
import numpy as np
import torch
from torch import nn
from torch import distributions

from typing import cast

results_dir = "/home/mohan/research/experiments/bc/"
demo_path = "/home/mohan/Downloads/1620492100_4904742/demo.hdf5"

flags.DEFINE_integer('amp_factor', 5, 'amplification factor')
flags.DEFINE_integer('amp_start', -80, 'start time step for range to be amplified.')
flags.DEFINE_integer('amp_end', -20, 'start time step for range to be amplified.')
flags.DEFINE_boolean('train', 1, 'whether to train a model or evaluate a model')
flags.DEFINE_integer('train_iterations', 250000, 'number of training iterations.')

import time

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

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation_tensor = torch.tensor(observation, dtype=torch.float).to(ptu.device)
        action_distribution = self.forward(observation_tensor)
        return cast(
            np.ndarray,
            action_distribution.sample().cpu().detach().numpy(),
        )[0]

FLAGS = flags.FLAGS

def main(_):
    start = time.time()

    # TODO: Save demonstrations in reverb
    demo_config = bc_robo_utils.DemoConfig(FLAGS.amp_factor, FLAGS.amp_start, FLAGS.amp_end, not FLAGS.train)
    dataset = bc_robo_utils.make_demonstrations(demo_path, demo_config)
    demo_time = time.time()

    print(f"demo took {demo_time-start} seconds.")

    obs, action = dataset[0][0], dataset[1][0]
    ob_dim = len(obs)
    ac_dim = len(action)
    n_layers = 4 # Change to 2
    size = 32 # Change to 300
    learning_rate = 0.01
    num_train_iterations = FLAGS.train_iterations
    batch_size = 32
    eval_steps = 250

    counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')

    print(f"len of generated dataset: {len(dataset[0])}")


    # Create the networks
    policy_network = PolicyNet(
                    ac_dim=ac_dim,
                    ob_dim=ob_dim,
                    n_layers=n_layers,
                    size=size)

    learner = bc.BCLearner(network=policy_network,
                        learning_rate=learning_rate,
                        dataset=dataset,
                        batch_size=batch_size, 
                        counter=learner_counter, 
                        logger=loggers.TerminalLogger('training', time_delta=0.))


    # get_action method should be move to a separate actor
    if FLAGS.train:
        for iter in range(num_train_iterations):
            learner.step()
        training_time = time.time()
        print(f"training took {training_time-demo_time} seconds.")

        print(f"saving the model at {results_dir}_{num_train_iterations}net.pt")
        learner.save(results_dir + f"{num_train_iterations}_net.pt")

    # TODO: Convert evaluation loop to an actor
    eval_policy_net = PolicyNet(
                    ac_dim=ac_dim,
                    ob_dim=ob_dim,
                    n_layers=n_layers,
                    size=size)
    eval_policy_net.load_state_dict(torch.load(results_dir + f"net.pt"))
    
    eval_policy_net.load_state_dict(torch.load(results_dir + f"{num_train_iterations}_net.pt"))

    eval_env = bc_robo_utils.make_eval_env(demo_path)
    # TODO: Move evaluation code to appropriate file
    full_obs = eval_env.reset()
    flat_obs = np.append(full_obs["robot0_proprio-state"], (full_obs["object-state"]))
    action = eval_policy_net.get_action(flat_obs)

    # TODO: Do eval after certain training steps
    for i in range(eval_steps):
        # act and observe
        obs, reward, done, _ = eval_env.step(action)
        eval_env.render()
        
        # compute next action
        flat_obs = np.append(full_obs["robot0_proprio-state"], (full_obs["object-state"]))
        action = eval_policy_net.get_action(flat_obs)
        if done:
            break 

if __name__ == '__main__':
  app.run(main)



