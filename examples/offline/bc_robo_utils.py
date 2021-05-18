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

def make_demonstrations(demo_path: Path) -> ndarray:
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
    
    transitions: List[Transition] = []

    # TODO: Decide how to batch transitions across episodes
    # Dataset is collected in the form of transitions.
    all_actions = []
    all_observations = []

    for i, episode in enumerate(tqdm(episodes[:1])):
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