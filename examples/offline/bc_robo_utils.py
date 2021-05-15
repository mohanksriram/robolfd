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


def make_demonstrations(demo_path: Path) -> ndarray:
    f = h5py.File(demo_path, "r")

    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    episodes = list(f["data"].keys())

    env = robosuite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )
    
    observations: List[Transition] = []

    # TODO: Decide how to batch observations across episodes
    # Dataset is collected in the form of transitions.
    for episode in episodes:
        # each demo is considered to be an episode

        model_xml = f[f"data/{episode}"].attrs["model_file"]
        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()

        states = f[f"data/{episode}/states"][()]
        actions = np.array(f[f"data/{episode}/actions"][()])

        for j, action in enumerate(actions):
            observation, reward, done, misc = env.step(action)
            trans = Transition()
            observations.append(Transition(observation, action, reward, None, None))
            if j > 0:
                # update o_tp1 for the previous observation
                observations[j-1].next_observation = observation
            
    return np.array(observations)
