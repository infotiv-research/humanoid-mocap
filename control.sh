#!/bin/bash
export ROS_DOMAIN_ID=$(cat ROS_DOMAIN_ID.txt)
echo "ROS_DOMAIN_ID.txt:" $ROS_DOMAIN_ID

input_dir="/home/ros/src/input"
output_dir="/home/ros/src/output"
image_file="$input_dir/image.png"
video_file="$input_dir/20250611video.mp4"
model_script="AUTOGLUON_model.py" # 

killz ()
{
    echo "::: KILLING PROCESSES  :::"
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
    pkill -9 -f eog
}
clean ()
{
    killz
    echo "::: DELETING AND RESTARTING THE PROJECT IN 5 SECONDS (including DATASET) :::"
    sleep 5

    rm -rf DATASET/motion_data/ DATASET/pose_data/  DATASET/pose_images/
    #rm -rf DATASET
    rm -rf ./build ./install ./log /output
    rm ROS_DOMAIN_ID.txt
    rm -rf /mlenv /models
    rm *.csv
}

build ()
{
    echo "::: BUILDING  :::"
    clean
    ROS_DOMAIN_ID=$((RANDOM % 102))
    echo $ROS_DOMAIN_ID > ROS_DOMAIN_ID.txt
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID
    colcon build --symlink-install  --packages-ignore  mlenv pose_to_motion
    echo "NEW ROS_DOMAIN_ID.txt:" $ROS_DOMAIN_ID
    mkdir -p "output"
    mlenv
}

mlenv ()
{
    python3 -m venv ~/mlenv
    source ~/mlenv/bin/activate
    pip install --upgrade pip
    pip install -r pose_to_motion/MLrequirements.txt
    deactivate
}

random ()
{
    echo "::: RANDOM  :::"
    source install/setup.bash ; ros2 launch random_motion_planner random_motion.launch.py output_dir:=$1 
}

camera ()
{
    echo "::: CAMERA  :::"
    source install/setup.bash ; ros2 run camera_system camera_viewer.py  --ros-args -p output_dir:=$1 
}
humanoid ()
{
    echo "::: HUMANOID  :::"
    source install/setup.bash ; ros2 launch humanoid_robot launch_humanoid.launch.py
}

clean_output ()
{
    echo "::: DELETE And rebuild /output :::"
    rm -rf output ; mkdir output ;
    mkdir output/motion_data 
    mkdir output/pose_data 
    mkdir output/pose_images 
}

replay_motion_helper ()
{
    echo "::: REPAYING $1 :::"
    source install/setup.bash ; ros2 launch random_motion_planner random_motion.launch.py motion_filename:="$1"
}

dataset ()
{
    echo "::: CREATING DATASET :::" in $1
    while true; do
        humanoid &
        sleep 5
        camera $1 &
        sleep 2
        random $1 &
        sleep 1200
        killz
        sleep 15
    done
}

if [[ "$*" == *"mlenv"* ]]
then
    mlenv
elif [[ "$*" == *"clean"* ]]
then
    clean
elif [[ "$*" == *"killz"* ]]
then
    killz
elif [[ "$*" == *"build"* ]]
then
    build
elif [[ "$*" == *"cmd"* ]]
then
    shift 1
    echo "RUNNING CMD ::: $@"
    exec "$@"
elif [[ "$*" == *"camera"* ]]
then
    camera
elif [[ "$*" == *"dataset"* ]]
then
    dataset $2
elif [[ "$*" == *"convert2csv"* ]]
then
    echo "::: Convert dataset to CSV, please wait :::"
    python3 pose_to_motion/load_json_data.py DATASET/EVAL/pose_data/  DATASET/EVAL/motion_data/   DATASET/eval.csv
    python3 pose_to_motion/load_json_data.py DATASET/TRAIN/pose_data/ DATASET/TRAIN/motion_data/  DATASET/train.csv
elif [[ "$*" == *"humanoid"* ]]
then
    humanoid
elif [[ "$*" == *"train"* ]]
then
    source ~/mlenv/bin/activate
    echo "TRAIN MODEL, OVERWRITING PREVIOUS MODEL"
    time python3 pose_to_motion/$model_script --mode train --filename DATASET/train.csv 2>&1 | tee training.log
elif [[ "$*" == *"eval"* ]]
then
    source ~/mlenv/bin/activate
    python3 pose_to_motion/$model_script --mode eval --filename DATASET/eval.csv
elif [[ "$*" == *"replay_motion"* ]]
then
    humanoid &
    sleep 5
    replay_motion_helper $2 # 2nd terminal argument

# ./control.sh predict DATASET/EVAL/
# ./control.sh predict output/
elif [[ "$*" == *"predict"* ]]  
then
    if [ "$#" -ne 2 ]; then
    echo "Error: The second argument is missing" >&2
    echo "Usage: $0 predict PATH_TO_DIR that has pose_data" >&2
    exit 1
    fi
    source ~/mlenv/bin/activate
    pose_dir="$2/pose_data" # input pose
    motion_dir="$2/motion_data" # output motion
    poseimage_dir="$2/pose_images" # pose ground truth image
    humanoid &
    sleep 5
    for file in $(ls $pose_dir | sort); do

        number="${file%%_*}"
        full_pose_path=$pose_dir/$number""_pose.json
        full_motion_path=$motion_dir/$number""_motion.json
        full_poseimage_path=$poseimage_dir/$number""_display.jpg
        echo ":::::::::::: GROUND TRUTH {$full_poseimage_path}" 
        if [ -f $full_poseimage_path ]; then 
            eog $full_poseimage_path & # open pose ground truth image
        else
            echo "NOT FOUND: $full_poseimage_path"
        fi
        echo "Processing: $file --- $number ----"
        python3 pose_to_motion/$model_script --mode predict --posefile $full_pose_path --motionfile "${full_motion_path}_PREDICTED"
        replay_motion_helper "${full_motion_path}_PREDICTED" &
        
        sleep 10
        pkill -9 -f motion_planner
        pkill -9 -f eog
    done
elif [[ "$*" == *"image_pipeline"* ]]
then
    clean_output
    python3 pose_to_motion/mp_detection.py --action image "$image_file" "output/"
elif [[ "$*" == *"video_pipeline"* ]]
then
    clean_output
    python3 pose_to_motion/mp_detection.py --action video "$video_file" "output/"
fi