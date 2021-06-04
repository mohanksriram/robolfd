from robolfd.agents.torch import bc
from robolfd.agents.torch import actors
from robolfd.torch.networks import MLP
from robolfd.utils import pytorch_util as ptu
from robolfd.utils import counting
from robolfd.utils import loggers

from examples.offline import bc_robo_utils

from absl import app
from absl import flags
import imageio
import numpy as np
import pathlib
import torch

import itertools
from typing import cast
import wandb

run = wandb.init(project="bc-panda-lift")

results_dir = "/home/mohan/research/experiments/bc/panda_lift/models/"
demo_path = "/home/mohan/research/experiments/bc/panda_lift/expert_demonstrations/1622106811_9832993/demo.hdf5"
traj_path = "/home/mohan/research/experiments/bc/panda_lift/trajectories/"
traj_fname = "trajectories.npy"


flags.DEFINE_integer('max_episodes', 100, 'maximum number of episodes to be used for training.')
flags.DEFINE_integer('num_actors', 10, 'number of actors enacting the demonstrations.')
flags.DEFINE_float('valid_pct', 0.2, 'percentage of episodes to be used for validation.')
flags.DEFINE_boolean('train', 1, 'whether to train a model or evaluate a model')

flags.DEFINE_integer('batch_size', 1024, 'batch size for training update.')
flags.DEFINE_integer('hidden_size', 128, 'dimension of each hidden layer.')
flags.DEFINE_float('lr', 5e-3, 'learning rate.')
flags.DEFINE_integer('n_layers', 2, 'number of hidden layers.')
flags.DEFINE_integer('train_iterations', 1000, 'number of training iterations.')

flags.DEFINE_float('evaluate_factor', 1/10, 'percentage of evaluations compared to train iterations.')
flags.DEFINE_float('log_factor', 1/100, 'percentage of logs compared to train iterations.')

flags.DEFINE_bool('cache_obs', False, 'whether to cache observations for reuse.')
flags.DEFINE_boolean('gpu', 1, 'whether to run on a gpu.')
flags.DEFINE_string('video_path', '/tmp/', 'where to store the rollouts.')
flags.DEFINE_list('obs_keys', ['robot0_proprio-state', 'object-state'],
                  'list of keys to include as part of the observation.')

import time

FLAGS = flags.FLAGS

def main(_):
    start = time.time()

    if FLAGS.gpu:
        ptu.init_gpu()

    # TODO: Save demonstrations in reverb
    demo_config = bc_robo_utils.DemoConfig(FLAGS.obs_keys, FLAGS.max_episodes, FLAGS.num_actors)
    
    # Check if observations are already part of a file
    file = pathlib.Path(traj_path + traj_fname)
    expert_trajectories = None
    if file.exists() and FLAGS.cache_obs:
        print("loading demonstrations from file")
        expert_trajectories = np.load(file, allow_pickle=True)
    else:
        print("generating demonstrations")
        expert_trajectories = bc_robo_utils.make_demonstrations(demo_path, demo_config)
        np.save(file, expert_trajectories, allow_pickle=True)

    valid_idx = int((1-FLAGS.valid_pct) * len(expert_trajectories))
    train_trajectories = expert_trajectories[:valid_idx]
    val_trajectories = expert_trajectories[valid_idx:]

    # Merge trajectories into transitions
    train_transitions = list(itertools.chain(*train_trajectories))
    val_transitions = list(itertools.chain(*val_trajectories))

    demo_time = time.time()
    print(f"demo took {demo_time-start} seconds.")
    
    print(f"len train_transitions: {len(train_transitions)}, val_transitions: {len(val_transitions)}")
    obs, action = train_transitions[0]
    print(obs)
    ob_dim = len(obs)
    ac_dim = len(action)
    print(f"obs dim: {ob_dim}, action dim: {ac_dim}")
    n_layers = FLAGS.n_layers # Change to 2
    size = FLAGS.hidden_size
    learning_rate = FLAGS.lr
    num_train_iterations = FLAGS.train_iterations
    batch_size = FLAGS.batch_size
    eval_steps = 250
 
    counter = counting.Counter()
    eval_counter = counting.Counter()
    learner_counter = counting.Counter(counter, prefix='learner')
    eval_learner_counter = counting.Counter(eval_counter, prefix='learner')

    # Create the networks
    policy_network = MLP(
                    in_dim=ob_dim,
                    out_dim=ac_dim,
                    n_layers=n_layers,
                    size=size)
    policy_network.train()
    
    learner = bc.BCLearner(network=policy_network,
                        learning_rate=learning_rate,
                        dataset=train_transitions,
                        batch_size=batch_size, 
                        counter=learner_counter, 
                        logger=loggers.TerminalLogger('training', time_delta=0.),
                        use_gpu=FLAGS.gpu)


    evaluate_every = int(num_train_iterations * FLAGS.evaluate_factor)
    log_every = int(num_train_iterations * FLAGS.log_factor)
    
    if FLAGS.train:
        # TODO: Convert evaluation loop to an agent
        eval_learner = bc.BCLearner(network=policy_network,
                            learning_rate=learning_rate,
                            dataset=val_transitions,
                            batch_size=batch_size, 
                            counter=eval_learner_counter, 
                            logger=loggers.TerminalLogger('validation', time_delta=0.),
                            use_gpu=FLAGS.gpu,
                            update_network=False
                            )
        eval_policy_net = MLP(
                        in_dim=ob_dim,
                        out_dim=ac_dim,
                        n_layers=n_layers,
                        size=size)
        eval_actor = actors.FeedForwardActor(eval_policy_net)
        eval_env = bc_robo_utils.make_eval_env(demo_path)
        
        for iter in range(num_train_iterations+1):
            train_loss_dict = learner.step()
            eval_loss_dict = eval_learner.step()
            
            if iter % log_every == 0:
                wandb.log({**train_loss_dict, **eval_loss_dict})

            if iter % evaluate_every == 0:
                model_checkpoint_name = f"{FLAGS.max_episodes}episodes__{iter}steps_{batch_size}bs_{FLAGS.hidden_size}hs_{n_layers}hl_net.pt"
                learner.save(results_dir + model_checkpoint_name)

                eval_policy_net.load_state_dict(torch.load(results_dir + model_checkpoint_name))
                eval_policy_net.eval()

                # TODO: Move evaluation code to appropriate file
                full_obs = eval_env.reset()
                flat_obs = np.concatenate([full_obs[key] for key in FLAGS.obs_keys])
                action = eval_actor.select_action(flat_obs)
                video_path = FLAGS.video_path + f"{FLAGS.max_episodes}episodes__{iter}steps_{batch_size}bs_{FLAGS.hidden_size}hs_{n_layers}hl_video.mp4"
                # create a video writer with imageio
                writer = imageio.get_writer(video_path, fps=20)

                for i in range(eval_steps):
                    # act and observe
                    obs, reward, done, _ = eval_env.step(action)
                    # compute next action
                    flat_obs = np.concatenate([obs[key] for key in FLAGS.obs_keys])
                    action = eval_actor.select_action(flat_obs)

                    # dump a frame from every K frames
                    if i % 1 == 0:
                        frame = obs["frontview_image"]
                        frame = np.flip(frame, 0)
                        writer.append_data(frame)

                    if done:
                        break 

        training_time = time.time()
        print(f"training took {training_time-demo_time} seconds.")
    run.finish()

if __name__ == '__main__':
  app.run(main)



