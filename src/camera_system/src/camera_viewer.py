#!/usr/bin/env python3
import os
from pathlib import Path
import json
import numpy as np
import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from camera_system.srv import CaptureMotionStatus


os.environ['MEDIAPIPE_MODEL_PATH'] = str(Path.home() / '.mediapipe/models')
os.makedirs(os.environ['MEDIAPIPE_MODEL_PATH'], exist_ok=True)

class PoseDetector(Node):
    def __init__(self):
        super().__init__('pose_detector')
        
        self.declare_parameter('pose_dir', 'DATASET/pose_data')  
        self.declare_parameter('image_dir', 'DATASET/pose_images')
        
        self.pose_dir = self.get_parameter('pose_dir').value
        self.image_dir = self.get_parameter('image_dir').value
        
        self.current_motion_id = None
        self.should_capture = False
        
        self.bridge = CvBridge()
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.01,
            min_tracking_confidence=0.01
        )
        
        self.latest_image = None
        # Subscribe to camera image
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',   
            self.store_image,
            10
        )
        
        # Service for synchronization with motion planner
        self.srv = self.create_service(CaptureMotionStatus, 'capture_motion', self.motion_status_callback)
        
        self.timer = self.create_timer(0.2, lambda: self.process_latest_image(save=False))

        self.get_logger().info(f'Detector node started with:')
        self.get_logger().info(f'- Output directory: {self.pose_dir}')
        self.get_logger().info(f'- Image directory: {self.image_dir}')

    def store_image(self, msg):
        # Store latest image read from camera
        self.latest_image = msg

    def motion_status_callback(self, request, response):
        """
        Handle motion capture requests from the motion planner
        If landmarks are detected: save pose data and image, return True
        If no landmarks are detected: return False        
        """
        motion_id = str(request.bodymotionid)
        self.get_logger().info(f"Received take picture for {motion_id}")
        self.process_latest_image(True)
        if self.results.pose_landmarks:
            self.save_pose_data(self.results.pose_landmarks, motion_id)
            # processed_image: without landmarks, display_image: with landmark
            self.save_pose_image(self.display_image, self.processed_image, motion_id)
            response.pose_data_capture = True
        else:
            response.pose_data_capture = False
        return response
              
    def process_latest_image(self, save = True):
            # Process latest image and pass it to MediaPipe, display the image if the pose data should be saved

            if self.latest_image is None:
                print("latest_image IS NONE")
                return
            print ("Processing")
            # Process every image from the camera
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "bgr8")
            if not save:
                self.pose.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            else:
                self.processed_image = cv_image
                self.results = self.pose.process(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
                # If pose landmarks are detected, display image
                if self.results.pose_landmarks:
                    self.display_image = self.processed_image.copy()
                    self.mp_drawing.draw_landmarks(
                        self.display_image,
                        self.results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )
                    cv2.imshow("Camera View", self.display_image)
                    cv2.waitKey(1)
    
    def save_pose_image(self, display_image, original_image, motionid):
        
        display_filename = f"{motionid}_display.jpg"
        display_path = os.path.join(self.image_dir, display_filename)
        cv2.imwrite(display_path, display_image)
        
        original_filename = f"{motionid}_original.jpg"
        original_path = os.path.join(self.image_dir, original_filename)
        cv2.imwrite(original_path, original_image)
        
        self.get_logger().info(f'Saved pose images to {self.image_dir}')

    def save_pose_data(self, pose_landmarks, motionid):
        # Save pose data as JSON file

        landmarks_data = []
        # Add the data for each detected landmark to landmarks_data
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks_data.append({
                'index': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        json_filename = f"{motionid}_data.json"
        json_path = os.path.join(self.pose_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(landmarks_data, f, indent=2)

        self.get_logger().info(f'Saved pose data to {json_path} ')

def main(args=None):
    rclpy.init(args=args)
    detector = PoseDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()