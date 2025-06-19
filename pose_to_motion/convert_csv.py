from loader import load_data
import numpy as np
import pandas as pd

if __name__ == "__main__":
    print("Loading data once at the beginning...")
    pose_dataset, motion_dataset = load_data()
    print(pose_dataset, motion_dataset )

    pose_num = pose_dataset.shape[1]
    motion_num = motion_dataset.shape[1]
    pose_col_names = [f'p{i+1}' for i in range(pose_num)]
    motion_col_names = [f'm{i+1}' for i in range(motion_num)]
    pose_df   = pd.DataFrame(pose_dataset, columns=pose_col_names)
    motion_df = pd.DataFrame(motion_dataset, columns=motion_col_names)
    result = pd.concat([pose_df, motion_df], axis=1).reindex(pose_df.index)
    result.to_csv('results.csv', index=True) 