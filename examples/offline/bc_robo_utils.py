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
    amp_factor: int
    amp_start: int
    amp_end: int
    last_episode_only: int
    max_episodes: int
    num_workers: int

    def __str__(self) -> str:
        return f"my config, amp_factor: {self.amp_factor}, amp_start: {self.amp_start}, amp_end: {self.amp_end}, last_episode_only: {self.last_episode_only}"

def generate_episode_transitions(demo_info):
    f, episode_num, config = demo_info
    # f = h5py.File(demo_path, "r")
    
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
    for j, action in enumerate(actions):
        observation, reward, done, misc = env.step(action)
        # use when you want to evaluate the environment
        # env.render()
        observations.append(observation)

    flat_observations = [np.append(observation["robot0_proprio-state"], (observation["object-state"]))
                            for observation in observations]

    # z
    all_observations.extend(flat_observations)
    all_actions.extend(actions)

    if config.amp_factor > 0:
        amplified_observations = config.amp_factor*flat_observations[-config.amp_start: -config.amp_end]
        amplified_actions = config.amp_factor*actions[-config.amp_start: -config.amp_end]

        # duplicate last few transitions to include more gripper closing action.
        all_observations.extend(amplified_observations)
        all_actions.extend(amplified_actions)

    return list(zip(all_observations, all_actions))


def make_demonstrations(demo_path: Path, config: DemoConfig) -> ndarray:
    f = h5py.File(demo_path, "r", skip_cache=False)

    episodes = list(f["data"].keys())[:config.max_episodes]
    episodes = episodes[-config.last_episode_only:]
    
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
        merged = list(itertools.chain(*transitions))
        return merged

def make_eval_env(demo_path: Path, has_offscreen_renderer = True):
    f = h5py.File(demo_path, "r")

    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    
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