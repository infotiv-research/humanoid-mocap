import torch
import numpy as np
import json
import os
from improvedModel import ImprovedMotionModel
from simpleModel import SimpleMotionModel
import loader
import random
import sys


NUM_JOINTS = 47
NUM_MARKERS = 33
np_motions = np.empty((0, NUM_JOINTS))
np_poses = np.empty((0, NUM_MARKERS * 3))

def test_model(model_path, test_pose_file, output_file):
   
    with open(test_pose_file, 'r') as f:
        pose_json = json.load(f)
    
    flattened_dict = loader.flatten_pose(pose_json)
    
    pose_data = np.append(np_poses, [np.array(list(flattened_dict.values()))], axis=0)
    pose_data_norm = loader.data_preprocess(pose_data)
    
    input_size = pose_data_norm.shape[1]
    output_size = NUM_JOINTS
    print("Input size, output size", input_size, output_size) 

    # Remember to change the model here if you want to try with a different one
    model = ImprovedMotionModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    input_tensor = torch.tensor(pose_data_norm, dtype = torch.float32)
    
    with torch.no_grad():
        prediction = model(input_tensor).squeeze(0).numpy()
    
    print("Saving prediction in " + output_file)
    loader.motion_to_json(prediction, output_file)

if __name__ == "__main__":
    model_path = sys.argv[1]
    random_file_path = sys.argv[2]
    output_file = sys.argv[3]
    
    test_model(model_path, random_file_path, output_file)

