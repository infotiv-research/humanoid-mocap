#!/bin/bash
export ROS_DOMAIN_ID=$(cat ROS_DOMAIN_ID.txt)
echo "ROS_DOMAIN_ID.txt:" $ROS_DOMAIN_ID

model_path="/home/ros/src/models/models/pose_motion_model.pth"
input_dir="/home/ros/src/input"
image_dir_pose="/home/ros/src/output"
image_file="$input_dir/999999999.png"
image_file_pose="$input_dir/999999999.png_data.json"
image_file_prediction="$image_dir_pose/999999999_prediction.json"
video_dir_pose="/home/ros/src/input/"
video_file="$video_dir_pose/20250611video.mp4"
video_dir_prediction="/home/ros/src/output/"


killz ()
{
    echo "--- killing processes ---"
    pkill -9 -f gzserver
    pkill -9 -f gzclient
    pkill -9 -f gazebo
    pkill -9 -f ign
    pkill -9 -f gz
    pkill -9 -f rviz

    pkill -9 -f camera_system
    pkill -9 -f motion_planner

    pkill -9 -f humble
    pkill -9 -f move_
    pkill -9 -f ros2
    pkill -9 -f ruby
}
clean ()
{
    killz
    rm -rf DATASET/motion_data/ DATASET/pose_data/  DATASET/pose_images/
    rm -rf ./build ./install ./log /output
    rm ROS_DOMAIN_ID.txt
}

build ()
{
    clean
    ROS_DOMAIN_ID=$((RANDOM % 102))
    echo $ROS_DOMAIN_ID > ROS_DOMAIN_ID.txt
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    colcon build --symlink-install ;
    echo "NEW ROS_DOMAIN_ID.txt:" $ROS_DOMAIN_ID
    mkdir -p "output"
}

view ()
{
    echo "::: VIEW  :::"
    source install/setup.bash ; ros2 run camera_system camera_viewer.py
}

random ()
{
    echo "::: RANDOM  :::"
    source install/setup.bash ; ros2 launch random_motion_planner random_motion.launch.py 
}


replay ()
{
    echo "replaying $1"
    source install/setup.bash ; ros2 launch random_motion_planner random_motion.launch.py motion_filename:="$1"
}

predict ()
{
    python3 pose_to_motion/test.py "$model_path" "$1" "$2"
}


image_pipeline () 
{
    echo "::: IMAGE PIPELINE :::"
    python3 pose_detection/mp_detection.py --action image "$image_file" "$image_dir_pose"

    sleep 2 
    humanoid & 
    sleep 5
    
    predict  "$image_file_pose" "$image_file_prediction"
    replay "$image_file_prediction"
}

video_pipeline () 
{
    echo "::: VIDEO PIPELINE :::"
    python3 pose_detection/mp_detection.py --action video "$video_file" "$video_dir_pose"
    sleep 5
    humanoid & 
    sleep 5
    for pose_file in "$video_dir_pose"/*_data.json; do
        if [[ -f "$pose_file" ]]; then
            pose_file_basename=$(basename "$pose_file")
            prediction_filename="$video_dir_prediction${pose_file_basename}_prediction.json"
            predict "$pose_file" "$prediction_filename"       
            echo "Completed processing: $pose_file"
            
            replay "$prediction_filename" &
            sleep 5
        else 
            echo "File $pose_file does not exist."
        fi
    done

    echo "::: COMPLETED PREDICTION AND MOTION REPLAY :::"
    wait 
}

humanoid ()
{
    echo "::: HUMANOID  :::"
    source install/setup.bash ; ros2 launch humanoid_robot launch_humanoid.launch.py
}


dataset ()
{
    while true; do
        humanoid &
        sleep 5
        view &
        sleep 5
        random &
        sleep 600
        killz
        echo "All processes are killed"
        sleep 30
    done
}

train_optuna ()
{
    python3 pose_to_motion/optimization_optuna.py /home/ros/src/DATASET/pose_data /home/ros/src/DATASET/motion_data
}

train ()
{
    echo "::: DATASET  :::"
    python3 pose_to_motion/train.py /home/ros/src/DATASET/pose_data /home/ros/src/DATASET/motion_data
}

symbolic ()
{
    echo "::: SYMBOLIC  :::"
    python3 /home/ros/src/pose_to_motion/Symbolicregresson.py /home/ros/src/DATASET/pose_data /home/ros/src/DATASET/motion_data
}

if [[ "$*" == *"clean"* ]]
then
    clean
elif [[ "$*" == *"build"* ]]
then
    build
elif [[ "$*" == *"cmd"* ]]
then
    shift 1
    echo "running ::: $@"
    exec "$@"
elif [[ "$*" == *"view"* ]]
then
    view
elif [[ "$*" == *"dataset"* ]]
then
    dataset
elif [[ "$*" == *"random"* ]]
then 
    random
elif [[ "$*" == *"predict"* ]]
then
    predict
elif [[ "$*" == *"humanoid"* ]]
then
    humanoid
elif [[ "$*" == *"train"* ]]
then
    train
elif [[ "$*" == *"train_optuna"* ]]
then
    train_optuna
elif [[ "$*" == *"image_pipeline"* ]]
then
    image_pipeline
elif [[ "$*" == *"video_pipeline"* ]]
then
    video_pipeline
elif [[ "$*" == *"symbolic"* ]]
then
    symbolic
fi