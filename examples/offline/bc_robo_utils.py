from dataclasses import dataclass
import h5pickle as h5py
import json
import numpy as np
from numpy import ndarray
from pathlib import Path

from typing import List
import random
from robolfd.types import Transition
import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

import itertools
from tqdm import tqdm
from multiprocessing import Pool

@dataclass
class DemoConfig:
    max_episodes: int
    num_workers: int

    def __str__(self) -> str:
        return f"my config, max_episodes: {self.max_episodes}, num_workers: {self.num_workers}"

def generate_episode_transitions(demo_info):
    f, episode_num, config = demo_info
    
    episodes = list(f["data"].keys())
    episode = episodes[episode_num]

    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    model_xml = f[f"data/{episode}"].attrs["model_file"]
    env.reset()
    xml = postprocess_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()

    all_observations = []
    all_actions = []
    
    # TODO: start from state
    states = f[f"data/{episode}/states"][()]

    actions = np.array(f[f"data/{episode}/actions"][()])

    # load the initial state
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    observations = []
    action = [0, 0, 0, -1]
    observation, _, _, _ = env.step(action)
    # observe the current state
    observations.append(observation)

    used_actions = []
    # Fix the order of action, observation sampling problem here
    for j, action in enumerate(actions):
        if not (action == [0, 0, 0, -1]).all():
            action = np.clip(action, -1, 1)
            observation, reward, done, misc = env.step(action)
            # use when you want to evaluate the environment
            # env.render()
            used_actions.append(action)
            observations.append(observation)
    # repeat last action for last observation
    used_actions.append(actions[-1])

    flat_observations = [np.concatenate((observation["robot0_eef_pos"], observation["robot0_eef_quat"], observation["robot0_gripper_qpos"], observation["object-state"]))
                            for observation in observations]

    # z
    all_observations.extend(flat_observations)
    all_actions.extend(used_actions)

    return list(zip(all_observations, all_actions))


def make_demonstrations(demo_path: Path, config: DemoConfig) -> ndarray:
    f = h5py.File(demo_path, "r", skip_cache=False)

    episodes = list(f["data"].keys())[:config.max_episodes]    
    # TODO: Decide how to batch transitions across episodes
    # Dataset is collected in the form of transitions.
    pbar = tqdm(total=len(episodes))


    with Pool(config.num_workers) as pool:
        # simple pool usage
        # transitions = pool.map(generate_episode_transitions, [(demo_path, i, config) for i in range(len(episodes))])
        # for measuring progress:
        res = [pool.apply_async(generate_episode_transitions, args=((f, i, config),),
                       callback=lambda _: pbar.update(1)) for i in range(len(episodes))]
        transitions = [p.get() for p in res]
        pool.close()
        pool.join()
        return transitions

def make_eval_env(demo_path: Path, robot_name="Panda", has_offscreen_renderer = True):
    f = h5py.File(demo_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])
    env_info['robots'] = robot_name
    
    env = robosuite.make(
        **env_info,
        has_renderer=not has_offscreen_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        ignore_done=True,
        use_camera_obs=has_offscreen_renderer,
        reward_shaping=True,
        control_freq=20,
        camera_names="frontview",
        camera_heights=512,
        camera_widths=512,
    )
    return env