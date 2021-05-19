from dataclasses import dataclass
import h5py
import json
import numpy as np
from numpy import ndarray
from pathlib import Path

from typing import List
import random
from robolfd.types import Transition
import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

from tqdm import tqdm

@dataclass
class DemoConfig:
    amp_factor: int
    amp_start: int
    amp_end: int
    last_episode_only: int

    def __str__(self) -> str:
        return f"my config, amp_factor: {self.amp_factor}, amp_start: {self.amp_start}, amp_end: {self.amp_end}, last_episode_only: {self.last_episode_only}"

def make_demonstrations(demo_path: Path, config: DemoConfig) -> ndarray:
    f = h5py.File(demo_path, "r")

    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    episodes = list(f["data"].keys())

    env = robosuite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    
    print(config)

    transitions: List[Transition] = []

    # TODO: Decide how to batch transitions across episodes
    # Dataset is collected in the form of transitions.
    all_actions = []
    all_observations = []

    for i, episode in enumerate(tqdm(episodes[-config.last_episode_only:])):
        # each demo is considered to be an episode

        model_xml = f[f"data/{episode}"].attrs["model_file"]
        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        states = f[f"data/{episode}/states"][()]
        actions = np.array(f[f"data/{episode}/actions"][()])

        observations = []
        for j, action in enumerate(actions):
            observation, reward, done, misc = env.step(action)
            observations.append(observation)

        flat_observations = [np.append(observation["robot0_proprio-state"], (observation["object-state"]))
                             for observation in observations]
        
        all_observations.extend(flat_observations)
        all_actions.extend(actions)

        amplified_observations = config.amp_factor*flat_observations[-config.amp_start: -config.amp_end]
        amplified_actions = config.amp_factor*actions[-config.amp_start: -config.amp_end]

        # duplicate last few transitions to include more gripper closing action.
        all_observations.extend(amplified_observations)
        all_actions.extend(amplified_actions)


    return (all_observations, all_actions)

def make_eval_env(demo_path: Path):
    f = h5py.File(demo_path, "r")

    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    return env