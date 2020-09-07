#!/usr/bin/env python
# coding: utf-8



import habitat_sim

import random
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import PyQt5
import os
import argparse

import numpy as np
import json

from pathlib import Path

parser = argparse.ArgumentParser(description="This script generates images for poses corresponding to given pose file. You have to add the `poses_run-replica-(scene_id).json` file beforehand in `GradGR-dataset-Habitat/data_collection/(scene_id)`. \n  If you're unsure about what you're doing, please set exp argument to 1. Setting 1 will only generate 5 images. Set 0 to generate all images.")
parser.add_argument("--scene_id", required=True, help = "scene id, for example: apartment_2")
parser.add_argument("--exp", required=True, help = "Are you experimenting? Set 1 if yes, 0 if no.")

args = parser.parse_args()


#TODO 1: Check below paths and assignments
# If you are using this for the first time/debugging, set the following to True. If you want to extract data corresponding to entire path in poses_*.json, then set False.
trial_data_collection = bool(int(args.exp)) 
# replica data and Data collection root paths
replica_data = "/scratch/shubodh/2020/Replica-Dataset/"
data_collection_root = '/scratch/shubodh/2020/GradGR-dataset-Habitat/data_collection/'
# Set scene path
scene_id = args.scene_id #assuming only Replica for now
test_scene = replica_data + scene_id + "/habitat/mesh_semantic.ply"
# Set existing poses file path (corresponding to which data will be extracted)
poses_json = data_collection_root + scene_id + "/poses_run-replica-" + scene_id + ".json"
# Whether you want to save a video correpsonding to robot trajectory, its path and name.
save_video = True
video_path = data_collection_root + scene_id + "/" 
video_name = 'data_run_video_' + scene_id
# Set to which paths you want to save raw data and visualization data  
raw_data_folder = data_collection_root + scene_id + "/raw_data/" 
viz_data_folder = data_collection_root + scene_id + "/viz_data/"

# Create paths if they don't exist
Path(raw_data_folder).mkdir(parents=True, exist_ok=True)
Path(viz_data_folder).mkdir(parents=True, exist_ok=True)


#TODO 2: Set settings like spatial resolution
sim_settings = {
    "width": 1920,  # Spatial resolution of the observations    
    "height": 1080,
    "scene": test_scene,  # Scene path
    "default_agent": 0,  
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": True,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}


# # Simulator config



def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]
    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },  
    }
    
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)
            
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])




cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# # Scene semantic annotations

def print_scene_recur(scene, limit_output=10):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
    
    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return None

# Print semantic annotation information (id, category, bounding box details) 
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene
print_scene_recur(scene)



from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

def display_save_sample(rgb_obs, semantic_obs, depth_obs, obs_id, save=True, visualize=False):
    # Save raw data
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    
    if save:
        rgb_img.save(raw_data_folder + str(obs_id) + "_rgb.png")
        
        depth_mm = depth_obs * 1000
        depth_mm_int = depth_mm.astype(np.int32)
        depth_pil = Image.fromarray(depth_mm_int, mode="I")
        depth_pil.save(raw_data_folder + str(obs_id) + '_depth_I.png')

        semantic_int = semantic_obs.astype(np.int32)
        semantic_pil = Image.fromarray(semantic_int, mode="I")
        semantic_pil.save(raw_data_folder + str(obs_id) + '_instance-seg_I.png')

    
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    # Save visualization data
    if save:
        rgb_jpg = rgb_img.convert("RGB")
        semantic_jpg = semantic_img.convert("RGB")
        depth_jpg = depth_img.convert("RGB")

        rgb_jpg.save(viz_data_folder + str(obs_id) + "_rgb.jpg")
        semantic_jpg.save(viz_data_folder + str(obs_id) + "_instance-seg.jpg")
        depth_jpg.save(viz_data_folder + str(obs_id) + "_depth.jpg")
        #print()
    # visualize first 3 frames
    if visualize and (obs_id < 3):
        arr = [rgb_img, semantic_img, depth_img]
        titles = ['rgb', 'semantic', 'depth']
        plt.figure(figsize=(12 ,8))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            plt.imshow(data, cmap='gray')
        plt.show()
    




#random.seed(sim_settings["seed"])
#sim.seed(sim_settings["seed"])

# Set agent state
with open(poses_json) as f:
    poses_data = json.load(f)
    
    position_array = [list(map(float,i)) for i in poses_data['position']]
    rotation_array = [list(map(float,i)) for i in poses_data['rotation']]



poses_path = {
    'position': np.array(position_array),
    'rotation': np.array(rotation_array)
}

agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = poses_path['position'][0]
agent_state.rotation = habitat_sim.utils.common.quat_from_coeffs(poses_path['rotation'][0])
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("AGENT'S INITIAL STATE: position", agent_state.position, "rotation", (agent_state.rotation))



total_frames = 0
action_names = list(
    cfg.agents[
        sim_settings["default_agent"]
    ].action_space.keys()
)

images_for_video = []


if trial_data_collection:
    max_frames = 5 
else:
    max_frames = len(poses_path['position'])

print("\n\nSAVING FIRST {} DATAPOINTS (RGB, DEPTH & INSTANCE) HAS STARTED. THE FOLLOWING ARE THE POSES CORRESPONDING TO WHICH DATA IS BEING EXTRACTED.\n\n".format(max_frames))
while total_frames < max_frames:
#     action = random.choice(action_names)
#     print("action", action)
    agent_state.position = poses_path['position'][total_frames]
    agent_state.rotation = habitat_sim.utils.common.quat_from_coeffs(poses_path['rotation'][total_frames])
    agent.set_state(agent_state)
    
    observations = sim.get_sensor_observations()
#     print((observations))
    rgb = observations["color_sensor"]
    semantic = observations["semantic_sensor"]
    depth = observations["depth_sensor"]
    
    images_for_video.append(rgb)
    
    display_save_sample(rgb, semantic, depth, total_frames, save=True, visualize=False)
    
    agent_state = agent.get_state()
    print("AGENT_STATE: position", agent_state.position, "rotation", agent_state.rotation)

    total_frames += 1


# Code for saving the video

import imageio
import tqdm
from habitat.core.logging import logger
from typing import Dict, List, Optional, Tuple


def img_to_video(images: List[np.ndarray],
                output_dir: str,
                video_name: str,
                fps: int = 30,
                quality: Optional[float] = 5,
                **kwargs,):
    
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()

if save_video:
    print(" \n \n VIDEO CORRESPONDING TO THE DATAPOINTS EXTRACTED IS BEING WRITTEN NOW --\n\n")
    img_to_video(images_for_video, video_path, video_name)




