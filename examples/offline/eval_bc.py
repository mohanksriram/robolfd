from robolfd.agents.torch import actors
from examples.offline import bc_robo_utils
from examples.offline.run_bc import FLAGS
from robolfd.torch.networks import MLP
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject

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
model_path = f"/home/mohan/research/experiments/bc/panda_lift/{experiment_name}/20episodes__1000steps_2048bs_128hs_2hl_net.pt"
video_path = "/home/mohan/research/experiments/bc/panda_lift/eval_rollouts/"

flags.DEFINE_integer('n_runs', 4, 'number of runs.')

FLAGS = flags.FLAGS

def main(_):
    start = time.time()
    robot_names = ['Panda']
    for robot_name in robot_names:
        # create evaluation environment
        # eval_env = bc_robo_utils.make_eval_env(demo_path, robot_name=robot_name)
        env_config = {
            "control_freq": 20,
            "env_name": "Lift",
            "hard_reset": False,
            "horizon": 500,
            "ignore_done": False,
            "reward_scale": 1.0,
            "camera_names": "frontview",
            "robots": [
            "Panda"
            ]
        }
        controller_config = load_controller_config(default_controller="OSC_POSITION")

        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        cube = BoxObject(
            name="cube",
            size_min=[0.020, 0.020, 0.020],  # [0.015, 0.015, 0.015],
            size_max=[0.022, 0.022, 0.022],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )
        table_offset = np.array((0, 0, 0.8))

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

        # x_runs = FLAGS.n_runs
        # y_runs = FLAGS.n_runs
        factor = 4/(FLAGS.n_runs*10)

        for x_run in tqdm(range(FLAGS.n_runs), "running..."):
            for y_run in range(FLAGS.n_runs):
                placement_initializer = UniformRandomSampler(
                    name="ObjectSampler",
                    mujoco_objects=cube,
                    x_range=[-0.22 + factor*x_run, -0.22 + factor*(x_run+1)],
                    y_range=[-0.22 + factor*y_run, -0.22 + factor*(y_run+1)],
                    rotation=None,
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=table_offset,
                    z_offset=0,
                )

                # eval_env = GymWrapper(eval_env)
                eval_env = suite.make(
                    **env_config,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    reward_shaping=True,
                    camera_heights=512,
                    camera_widths=512,
                    controller_configs=controller_config,
                    placement_initializer=placement_initializer
                )
                # run and save the eval_rollouts
                full_obs = eval_env.reset()
                flat_obs = np.concatenate([full_obs[key] for key in obs_keys])
                action = eval_actor.select_action(flat_obs)

                cur_path = video_path + f"eval_run_{experiment_name}_{robot_name}_{x_run}_{y_run}.mp4"
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

