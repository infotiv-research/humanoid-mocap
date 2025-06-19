import os
import glob
import json
import numpy as np
import sys

NUM_JOINTS = 47
NUM_MARKERS = 33
np_motions = np.empty((0, NUM_JOINTS))
np_poses = np.empty((0, NUM_MARKERS * 3))

def load_data():
    '''
    Loads .json data from pose and motion directories
    Ensures the motion ID in the filename matches between pose and motion
    Flattens the pose data to single dict from a list of dicts
    Returns both pose and motion data as numpy arrays
    '''
    global np_poses, np_motions
 
    pose_dir = sys.argv[1]
    motion_dir = sys.argv[2]

    # Look for .json files
    for filename in os.listdir(pose_dir):
        # Check if it's a file (not a directory)
        if os.path.isfile(os.path.join(pose_dir, filename)) :
            id = filename.split("_")[0]
            # print(id)
            try:
                posefile=os.path.join(pose_dir  ,id + '_data.json')
                motifile=os.path.join(motion_dir,id + '_motion.json')
                pdata = open(posefile, 'r')
                mdata = open(motifile, 'r')
                
                posedata = json.load(pdata)
                motidata = json.load(mdata)
            except:
                print("WARNING, MISSING FILE for ", id)

            fl_posedata= flatten_pose(posedata)
            np_poses = np.append(np_poses, [np.array(list(fl_posedata.values()))], axis=0)
            np_motions = np.append(np_motions, [np.array(list(motidata.values()))], axis=0)

    print(np_poses.shape)
    print(np_motions.shape)
    normalize_pose = data_preprocess(np_poses)

    return (normalize_pose, np_motions)

def data_preprocess(pose_dataset):

    landmark_index = 23
    ############POSE NORMALIZATION############
    normalized = pose_dataset - np.tile(pose_dataset[:, landmark_index*3:landmark_index*3 + 3], (1, 33))
    ##############HEIGHT NORMALIZATION##############
    # hight_index = 0
    # reference = pose_dataset[:, hight_index * 3: hight_index * 3 + 3]
    # reference = np.where(reference == 0, 1e-8, reference)
    # hight_norm = pose_dataset / np.tile(reference, (1, 33)) 
    # normalized = hight_norm - np.tile(hight_norm[:, landmark_index*3:landmark_index*3 + 3], (1, 33))
    ##############LANDMARK DECREASE##############
    # keep_landmarks = [i for i in range(33)] 
    # # keep_landmarks = [i for i in range(33) if i < 1 or i > 10]  # 0, 11–32
    # keep_indices = []
    # for i in keep_landmarks:
    #     keep_indices.extend([i * 3, i * 3 + 1, i * 3 + 2])  # x, y, z

    # filtered = normalized[:, keep_indices]



    return normalized

def flatten_pose(list_of_dicts):

    flattened_dict = {}
 
    for idx, d in enumerate(list_of_dicts):
        for key, value in d.items():
            if "index" not in key and "visibility" not in key:
                new_key = f"{idx}_{key}"  # Combine index and key
                flattened_dict[new_key] = value
    
    return flattened_dict

def motion_to_json(motions, json_file):
    '''
    Saves a motion as .json file
    '''

    joint_names = ["link1_link2_joint", "jT9T8_rotz", "jC7LeftShoulder_rotx", "jLeftShoulder_rotz", "jLeftElbow_roty", "jLeftWrist_rotx", "jRightShoulder_rotx", "jRightBallFoot_roty",
    "jT1C7_rotx", "jRightAnkle_rotx", "jRightAnkle_roty", "jRightWrist_rotx", "jLeftShoulder_rotx", "jLeftShoulder_roty", "jRightShoulder_roty", "jL5S1_roty", "jLeftAnkle_rotx",
    "jLeftKnee_roty", "jRightHip_rotz", "link3_link4_joint", "jT9T8_roty", "jRightWrist_rotz", "jLeftHip_roty", "jLeftAnkle_rotz","link2_link3_joint", "jLeftKnee_rotz", "jLeftWrist_rotz", 
    "jLeftAnkle_roty", "jT9T8_rotx", "jLeftElbow_rotz", "jLeftBallFoot_roty", "jRightKnee_rotz", "jLeftHip_rotx","jRightKnee_roty", "jRightHip_rotx", "jL5S1_rotx", "jC1Head_roty", 
    "jRightElbow_roty", "jC1Head_rotx", "jT1C7_rotz", "jRightAnkle_rotz", "jRightHip_roty", "jT1C7_roty", "jLeftHip_rotz", "jRightElbow_rotz", "jC7RightShoulder_rotx", "jRightShoulder_rotz"
    ]
    data = dict(zip(joint_names, motions.tolist()))
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    load_data()



