from robolfd.agents.torch import actors
from examples.offline import bc_robo_utils
from examples.offline.run_bc import FLAGS
from robolfd.torch.networks import MLP

from absl import app
from absl import flags
import imageio
import numpy as np
import torch

import time
from tqdm import tqdm
from typing import cast

experiment_name="models"

demo_path = "/home/mohan/research/experiments/bc/panda_lift/expert_demonstrations/1623322082_6477468/demo.hdf5"
model_path = f"/home/mohan/research/experiments/bc/panda_lift/{experiment_name}/75episodes__3000steps_2048bs_128hs_2hl_net.pt"
video_path = "/home/mohan/research/experiments/bc/panda_lift/eval_rollouts/"

flags.DEFINE_integer('n_runs', 4, 'number of runs.')

FLAGS = flags.FLAGS

def main(_):
    start = time.time()
    robot_names = ['Panda']
    for robot_name in robot_names:
        # create evaluation environment
        eval_env = bc_robo_utils.make_eval_env(demo_path, robot_name=robot_name)
        # eval_env = GymWrapper(eval_env)

        # create evaluation network
        eval_policy_net = MLP(
                        in_dim=42,
                        out_dim=4, # should be changed according to experiment
                        n_layers=2,
                        size=128)

        # load saved model
        eval_policy_net.load_state_dict(torch.load(model_path))
        eval_policy_net.eval()
        eval_actor = actors.FeedForwardActor(eval_policy_net)

        obs_keys = ['robot0_proprio-state', 'object-state']

        print(f"model loaded successfully")
        eval_steps = 250
        video_path = "/home/mohan/research/experiments/bc/panda_lift/eval_rollouts/"

        for run in tqdm(range(FLAGS.n_runs), "running..."):
            # run and save the eval_rollouts
            full_obs = eval_env.reset()
            flat_obs = np.concatenate([full_obs[key] for key in obs_keys])
            action = eval_actor.select_action(flat_obs)

            cur_path = video_path + f"eval_run_{experiment_name}_{robot_name}_{run}.mp4"
            # create a video writer with imageio
            writer = imageio.get_writer(cur_path, fps=20)

            for i in range(eval_steps):
                # act and observe
                obs, reward, done, _ = eval_env.step(action)
                # eval_env.render()
                # compute next action
                flat_obs = np.concatenate([obs[key] for key in obs_keys])
                action = eval_actor.select_action(flat_obs)

                # dump a frame from every K frames
                if i % 1 == 0:
                    frame = obs["frontview_image"]
                    frame = np.flip(frame, 0)
                    writer.append_data(frame)
                if done:
                    break
    
    end = time.time()
    print(f"total time taken: {end-start}s")

if __name__ == '__main__':
  app.run(main)

