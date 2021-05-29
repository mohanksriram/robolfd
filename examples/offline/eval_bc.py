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
import imageio
import numpy as np
import pathlib
import torch
from torch import nn
from torch import distributions

from examples.offline.run_bc import FLAGS, PolicyNet
import time
from tqdm import tqdm
from typing import cast

from robosuite.wrappers import GymWrapper

experiment_name="models_only_object_state"

demo_path = "/home/mohan/research/experiments/bc/panda_lift/expert_demonstrations/1621926034_0051796/demo.hdf5"
model_path = f"/home/mohan/research/experiments/bc/panda_lift/{experiment_name}/40episodes__2500steps_10bs_100hs_2hl_net.pt"
video_path = "/home/mohan/research/experiments/bc/panda_lift/eval_rollouts/"

flags.DEFINE_integer('n_runs', 2, 'number of runs.')

FLAGS = flags.FLAGS

def main(_):
    start = time.time()
    robot_names = ['Panda', 'Sawyer', 'IIWA', 'Jaco', 'Kinova3', 'UR5e', 'Baxter']
    for robot_name in robot_names:
        # create evaluation environment
        eval_env = bc_robo_utils.make_eval_env(demo_path, robot_name=robot_name)
        # eval_env = GymWrapper(eval_env)

        # create evaluation network
        eval_policy_net = PolicyNet(
                        ac_dim=7,
                        ob_dim=10, # should be changed according to experiment
                        n_layers=2,
                        size=100)

        # load saved model
        eval_policy_net.load_state_dict(torch.load(model_path))
        eval_policy_net.eval()


        print(f"model loaded successfully")
        eval_steps = 250
        video_path = "/home/mohan/research/experiments/bc/panda_lift/eval_rollouts/"

        for run in tqdm(range(FLAGS.n_runs), "running..."):
            # run and save the eval_rollouts
            full_obs = eval_env.reset()
            flat_obs = full_obs["object-state"]
            action = eval_policy_net.get_action(flat_obs)

            cur_path = video_path + f"eval_run_{experiment_name}_{robot_name}_{run}.mp4"
            # create a video writer with imageio
            writer = imageio.get_writer(cur_path, fps=20)

            for i in range(eval_steps):
                # act and observe
                obs, reward, done, _ = eval_env.step(action)
                # eval_env.render()
                # compute next action
                flat_obs = full_obs["object-state"]
                action = eval_policy_net.get_action(flat_obs)

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

