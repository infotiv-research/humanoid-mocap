#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2, MoveIt2State
import random
import time
import json
import os
from camera_system.srv import CaptureMotionStatus
import random


class WholeBodyMotionPlanner(Node):
    def __init__(self, filename=None):
        super().__init__('random_motion_planner')

        self.declare_parameter('motion_filename', None)
        motion_filename = self.get_parameter('motion_filename').get_parameter_value().string_value

        # Client for camera synchronization, only for random motion
        if motion_filename == "random" :
            self.cli = self.create_client(CaptureMotionStatus, 'capture_motion')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for service...')
            self.req = CaptureMotionStatus.Request()

        self.declare_parameter('max_velocity', 0.9)      
        self.declare_parameter('max_acceleration', 0.9)  
        self.declare_parameter('motion_amplitude', 1.0) 
        self.declare_parameter('planner_id', 'RRTConnectkConfigDefault')

        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.motion_amplitude = self.get_parameter('motion_amplitude').value
        self.planner_id = self.get_parameter('planner_id').value

        self.declare_parameter('pose_dir', 'DATASET/pose_data')  
        self.declare_parameter('image_dir', 'DATASET/pose_images')
        self.declare_parameter('motion_dir', 'DATASET/motion_data')
        self.pose_dir = self.get_parameter('pose_dir').value
        self.image_dir = self.get_parameter('image_dir').value
        self.motion_dir = self.get_parameter('motion_dir').value
        
        os.makedirs(self.pose_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.motion_dir, exist_ok=True)
        
        self.movable_joints = [
           'jL5S1_rotx', 'jL5S1_roty','jT9T8_rotx', 'jT9T8_roty', 'jT9T8_rotz',
            'jC7LeftShoulder_rotx', 'jLeftShoulder_rotx','jLeftShoulder_roty', 'jLeftShoulder_rotz',
            'jLeftElbow_roty', 'jLeftElbow_rotz','jLeftWrist_rotx', 'jLeftWrist_rotz',
            'jC7RightShoulder_rotx', 'jRightShoulder_rotx', 'jRightShoulder_roty', 'jRightShoulder_rotz',
            'jRightElbow_roty', 'jRightElbow_rotz','jRightWrist_rotx', 'jRightWrist_rotz',
            'jRightHip_rotx', 'jRightHip_roty', 'jRightHip_rotz', 'jRightKnee_roty', 'jRightKnee_rotz', 
            'jRightAnkle_rotx', 'jRightAnkle_roty', 'jRightAnkle_rotz', 'jRightBallFoot_roty',
            'jLeftHip_rotx', 'jLeftHip_roty', 'jLeftHip_rotz', 'jLeftKnee_roty', 'jLeftKnee_rotz', 
            'jLeftAnkle_rotx', 'jLeftAnkle_roty', 'jLeftAnkle_rotz', 'jLeftBallFoot_roty',
            'jT1C7_rotx', 'jT1C7_roty', 'jT1C7_rotz', 'jC1Head_rotx', 'jC1Head_roty',
            'link1_link2_joint', 'link2_link3_joint', 'link3_link4_joint'
        ]
        
        # Set all joints to 0.0 as initial position
        self.current_joint_positions = {joint: 0.0 for joint in self.movable_joints}
        
        self.current_motion = None
        self.current_motion_id = None
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.movable_joints,
            group_name='support_whole_body',
            base_link_name='Pelvis',
            end_effector_name='',
            use_move_group_action=True,
        )
        
        self.moveit2.planner_id = self.planner_id
        self.moveit2.max_velocity = self.max_velocity
        self.moveit2.max_acceleration = self.max_acceleration
        
        time.sleep(1.0)

        if motion_filename == "random" :
            # RANDOM MOTION
            self.update_current_joint_positions()
            
            self.get_logger().info('Starting in synchronized mode with camera')

            seed_value = int(time.time())
            random.seed(seed_value)

            for i in range(10000):
                self.current_motion_id = random.randint(1, 9999999999)
                self.plan_and_execute_motion()
                time.sleep(0.5)
        else:
            # Execute motion from file via MoveIt2
            with open(motion_filename, 'r') as file:
                data = json.load(file)
            joint_names = list(data.keys())
            joint_positions = list(data.values())

            self.get_logger().info('Executing motion from file')

            self.moveit2.move_to_configuration(
                joint_positions,
                joint_names
            )

    def get_joint_state_dict(self, joint_state=None):
        # Read joint state and return dict of joint names and positions
  
        if joint_state is None:
            joint_state = self.moveit2.joint_state
            if joint_state is None:
                return None
            
        joint_positions = {}
        for i, name in enumerate(joint_state.name):
            joint_positions[name] = joint_state.position[i]
        
        return joint_positions

    def update_current_joint_positions(self):
        # Initialization of joint positions: Get current joint positions as dict and update current joint positions 
           
        joint_state_dict = self.get_joint_state_dict()
        if joint_state_dict:
            for joint in self.movable_joints:
                if joint in joint_state_dict:
                    self.current_joint_positions[joint] = joint_state_dict[joint]
            self.get_logger().info('Current joint positions updated from robot state')
        else:
            self.get_logger().warn('Could not get current joint state, using default values')

    def log_error_message(self):
        if hasattr(self.moveit2, 'get_last_execution_error_code') and self.moveit2.get_last_execution_error_code() is not None:
            self.get_logger().warn(f'Motion {self.current_motion_id} Motion execution ended with state: {self.moveit2.query_state()}')                    
            self.get_logger().info(f'Motion {self.current_motion_id} Retrying with a new motion after abort')

    def save_current_motion(self, motion_id):
        # Save current motion as [motion ID]_motion.json

        if not self.current_motion:
            self.get_logger().warn("No motion data to save")
            return

        json_filename = os.path.join(self.motion_dir, f"{motion_id}_motion.json")
        with open(json_filename, 'w') as f:
            json.dump(self.current_motion, f, indent=2)
        
        self.get_logger().info(f"Motion sequence saved to {json_filename}")
        
        self.current_motion = None

    def send_request(self, motion_id):
        # Send a request to the camera viewer service server to capture the pose for the current motion

        self.req.bodymotionid = motion_id
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        r = future.result().pose_data_capture
        
        return r

    def plan_and_execute_motion(self):
            '''
            Read the current joint state
            Create a target joint position (Select offset or absolute value)
            Execute motion via MoveIt2
            If motion is successful, request camera capture and save motion data 
            '''
            
            # Get joint state before motion
            before_joint_state = self.get_joint_state_dict()
            if before_joint_state:
                self.get_logger().info(f"Before motion - Got state for {len(before_joint_state)} joints")
                self.current_joint_positions = before_joint_state
            
            joint_positions = []
            joint_names = []
            
            # Create target positions:
            # OFFSET OPTION: Add a random offset to current joint positions
            # NORMAL ABSOLUTE VALUE: Set a new target position from a normal distribution
            target_positions = {}

            for joint_name in self.movable_joints:
                ######## OFFSET OPTION ########
                current_val = self.current_joint_positions.get(joint_name, 0.0)
                random_offset = random.uniform(-self.motion_amplitude, self.motion_amplitude)
                target_val = current_val + random_offset
                
                # ######## NORMAL ABSOLUTE VALUE ########
                # random_offset = random.gauss(0, self.motion_amplitude / 8)
                # target_val = random_offset

                joint_positions.append(target_val)
                joint_names.append(joint_name)
                target_positions[joint_name] = target_val
                
            self.get_logger().info(f' Planning and executing for {len(joint_names)} joints')
            
            # Execute motion via MoveIt2
            self.moveit2.move_to_configuration(
                joint_positions,
                joint_names
            )
            
            # Wait until motion is executed (returns self.motion_suceeded true/false)
            success = self.moveit2.wait_until_executed()
            self.get_logger().info(f'Motion id = {self.current_motion_id}')

            if success:
                self.get_logger().info(f'Motion executed successfully')
                after_joint_state = self.get_joint_state_dict()
                if after_joint_state:
                    self.get_logger().info(f"After motion - Got state for {len(after_joint_state)} joints")
                    self.get_logger().info(f"Waiting for camera to capture pose data")
                    save = self.send_request(self.current_motion_id)
                    if save:                        
                        self.current_motion = after_joint_state
                        self.save_current_motion(self.current_motion_id)
                    else:
                        self.get_logger().info(f"Pose not detected")
                else:
                    self.get_logger().warn(f' Motion succeeded but could not get final state')

            else:
                # If motion did not succeed
                self.get_logger().info(f'Invalid motion')
                self.log_error_message()


def main(args=None):
    rclpy.init(args=args)
    planner = WholeBodyMotionPlanner()
    
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()