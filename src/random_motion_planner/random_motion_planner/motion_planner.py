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
import sys
random.seed(int(time.time()))
NUM_MOTION = 10000
sys.path.append('.')
import humanoid_config

class WholeBodyMotionPlanner(Node):
    def __init__(self, filename=None):
        super().__init__('random_motion_planner')

        self.declare_parameter('motion_filename', "DEFAULT_MOTION_FILENAME")
        motion_filename = self.get_parameter('motion_filename').get_parameter_value().string_value

        self.declare_parameter('output_dir', 'DATASET')
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value

        self.declare_parameter('motion_dir', self.output_dir + '/motion_data')
        self.motion_dir = self.get_parameter('motion_dir').value
        os.makedirs(self.motion_dir, exist_ok=True)
        self.get_logger().info(f"Motion directory: {self.motion_dir}")

        # Client for camera synchronization, only for random motion
        if motion_filename == "random" :
            self.cli = self.create_client(CaptureMotionStatus, 'capture_motion')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('Waiting for service...')
            self.req = CaptureMotionStatus.Request()

        self.declare_parameter('max_velocity', 0.8)      
        self.declare_parameter('max_acceleration', 0.8)  
        self.declare_parameter('motion_amplitude', 0.8) 
        self.joint_limit = 0.8  # TO AVOID FULL RANGE 
        self.declare_parameter('planner_id', 'RRTConnectkConfigDefault')

        self.max_velocity = self.get_parameter('max_velocity').value
        self.max_acceleration = self.get_parameter('max_acceleration').value
        self.motion_amplitude = self.get_parameter('motion_amplitude').value
        self.planner_id = self.get_parameter('planner_id').value
        
        self.movable_joint_names = humanoid_config.movable_joint_names
        self.movable_joint_upper = humanoid_config.movable_joint_upper
        self.movable_joint_lower = humanoid_config.movable_joint_lower
        # Set all joints to 0.0 as initial position
        self.movable_joint_positions =  [0] *  len(self.movable_joint_names)
        
        self.current_motion = None
        self.current_motion_id = 9999
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.movable_joint_names,
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
            self.get_logger().info('Random Motion')
            for i in range(NUM_MOTION):
                self.current_motion_id = random.randint(1000000000, 9999999999)
                self.random_joint_positions()
                self.plan_and_execute_motion()
        else:
            self.get_logger().info('Executing motion from file via MoveIt2')
            with open(motion_filename, 'r') as file:
                data = json.load(file)
            joint_names = list(data.keys())
            joint_positions = list(data.values())
           

            self.moveit2.move_to_configuration(
                joint_positions,
                joint_names
            )

    def random_joint_positions(self):
        for i in range(len(self.movable_joint_names)):
            self.movable_joint_positions[i] = random.uniform(self.movable_joint_lower[i] * self.joint_limit, self.movable_joint_upper[i]*self.joint_limit)


    def log_error_message(self):
        if hasattr(self.moveit2, 'get_last_execution_error_code') and self.moveit2.get_last_execution_error_code() is not None:
            self.get_logger().info(f'Motion {self.current_motion_id} Motion execution ended with state: {self.moveit2.query_state()} {self.moveit2.get_last_execution_error_code()}')                    
            self.get_logger().info(f'Motion {self.current_motion_id} Retrying with a new motion after abort')

    def save_current_motion(self):
        # Save current motion as [motion ID]_motion.json
        current_motion  = dict(zip(self.moveit2.joint_state.name, self.moveit2.joint_state.position)) 
        json_filename = os.path.join(self.motion_dir, f"{self.current_motion_id}_motion.json")
        with open(json_filename, 'w') as f:
            json.dump(current_motion, f, indent=2)
        self.get_logger().info(f"Motion sequence saved to {json_filename}")

    def send_request_camera(self):
        # Send a request to the camera viewer service server to capture the pose for the current motion
        self.req.bodymotionid = self.current_motion_id
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        r = future.result().pose_data_capture
        return r

    def plan_and_execute_motion(self):
            # Execute motion via MoveIt2
            self.moveit2.move_to_configuration(
                self.movable_joint_positions,
                self.movable_joint_names
            )
            # Wait until motion is executed (returns self.motion_succeeded true/false)
            success = self.moveit2.wait_until_executed()
            if success:
                self.get_logger().info(f'Motion is successful')
                #self.get_logger().info(f'Motion executed successfully')
                #self.get_logger().info(f'SENT:')
                #self.get_logger().info(str(dict(zip(self.movable_joint_names, self.movable_joint_positions)) ))
                #self.get_logger().info(f'STATE:')
                #self.get_logger().info(str(dict(zip(self.moveit2.joint_state.name, self.moveit2.joint_state.position)) ))
                req = self.send_request_camera()
                if req:                        
                    self.save_current_motion()
                else:
                    self.get_logger().info(f"Pose not detected")
            else:
                self.get_logger().info(f'Motion is NOT successful')
                self.log_error_message()
            time.sleep(0.2)


def main(args=None):
    rclpy.init(args=args)
    planner = WholeBodyMotionPlanner()
    
    rclpy.spin(planner)
    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()