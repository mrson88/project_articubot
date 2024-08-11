# #!/usr/bin/env python3
# import os
# import cv2
# from ultralytics import YOLO
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image,CameraInfo
# from cv_bridge import CvBridge
# from ament_index_python.packages import get_package_share_directory
# from articubot_msgs.msg import InferenceResult
# from articubot_msgs.msg import Yolov8Inference
# from geometry_msgs.msg import Twist
# bridge = CvBridge()
# import time
# from std_msgs.msg import Bool, String
# from geometry_msgs.msg import Point
# from articubot_msgs.action import ArticubotTask
# from rclpy.action import ActionClient
# from rclpy.action.client import ClientGoalHandle, GoalStatus
# # from Facerec_QT.face_rec import FaceRecognition,read_db,write_db
# import numpy as np
# from geometry_msgs.msg import PointStamped, PoseStamped
# from tf_transformations import quaternion_from_euler
# import torch
# from geometry_msgs.msg import TransformStamped
# from cv_bridge import CvBridge
# from tf2_ros import TransformBroadcaster
# class Camera_subscriber(Node):

#     def __init__(self):
#         super().__init__('camera_subscriber')
#         self.package_share_dir = get_package_share_directory("robot_recognition")
#         self.model_dir = os.path.join(self.package_share_dir, "scripts","yolov8m.pt")
#         # self.quantize_model_dir = os.path.join(self.package_share_dir, "scripts","yolov8n_quantized.pt")
#         if torch.cuda.is_available():
#             self.device=torch.device("cuda")
#         else:
#             self.device=torch.device("cpu")

#         self.model = YOLO(self.model_dir).to(self.device)
#         # self.state_dict = torch.load(self.quantize_model_dir)
#         # self.quantized_model = torch.quantization.quantize_dynamic(
#         #     self.model.model,
#         #     {torch.nn.Linear, torch.nn.Conv2d},
#         #     dtype=torch.qint8
#         # )
        
#         # # Load the quantized state dict
#         # self.quantized_model.load_state_dict(self.state_dict)
        
#         # # Replace the model in the YOLO object
#         # self.model.model = self.quantized_model

#         self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.yolov8_inference = Yolov8Inference()

#         self.subscription = self.create_subscription(
#             Image,
#             'camera/camera/color/image_raw',
#             self.camera_callback,
#             1)
#         # self.subscription = self.create_subscription(
#         #     Image,
#         #     '/image',
#         #     self.webcam_callback,
#         #     1)
#         self.subscription 
#         self.subscription_depth = self.create_subscription(
#             Image,
#             'camera/camera/depth/image_rect_raw',
#             self.depth_camera_callback,
#             1)
#         self.create_subscription(
#             CameraInfo,
#             'camera/camera/depth/camera_info',
#             self.camera_info_callback,
#             10)
#         self.publisher_point = self.create_publisher(PointStamped, 'point_3d', 5)
#         self.subscription_depth
#         self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 5)
#         self.img_pub = self.create_publisher(Image, "/inference_result", 5)
#         self._action_client = ActionClient(self, ArticubotTask, 'test_server')
#         self.sub_supress = self.create_subscription(String, 'find_ball', self.findball_callback, 5)
#         timer_period = 0.1  # seconds
#         self.timer = self.create_timer(timer_period, self.timer_callback)
#         self.target_val = 0.0
#         self.target_dist = 0.0
#         self.lastrcvtime = time.time() - 10000
#         self.pixel_x=0
#         self.pixel_y=0
#         self.detect=False
#         self.camera_info = None

#         self.declare_parameter("rcv_timeout_secs", 1.0)
#         self.declare_parameter("angular_chase_multiplier", 0.2)
#         self.declare_parameter("forward_chase_speed", 0.1)
#         self.declare_parameter("search_angular_speed", 0.15)
#         self.declare_parameter("max_size_thresh", 0.1)
#         self.declare_parameter("filter_value", 0.9)
#         self.rcv_timeout_secs = self.get_parameter('rcv_timeout_secs').get_parameter_value().double_value
#         self.angular_chase_multiplier = self.get_parameter('angular_chase_multiplier').get_parameter_value().double_value
#         self.forward_chase_speed = self.get_parameter('forward_chase_speed').get_parameter_value().double_value
#         self.search_angular_speed = self.get_parameter('search_angular_speed').get_parameter_value().double_value
#         self.max_size_thresh = self.get_parameter('max_size_thresh').get_parameter_value().double_value
#         self.filter_value = self.get_parameter('filter_value').get_parameter_value().double_value
#         self.inference_result = InferenceResult()
#         self.recognition_on=True
#         self.registration_data = None
#         self.frame_height = 480
#         self.frame_width = 640
#         self.depth_image=[]
#         # self.face_recognition = FaceRecognition(0.7, self.frame_height, self.frame_width)
#         self.findball = True

#     def camera_info_callback(self, msg):
#         self.camera_info = msg

#     def timer_callback(self):
#         msg = Twist()
#         if self.findball:
#             # if (time.time() - self.lastrcvtime < self.rcv_timeout_secs):
#             if self.inference_result.class_name=="sports ball" or self.inference_result.class_name=="frisbee" :
#                 # self.get_logger().info('Target: {}={}'.format(self.inference_result.class_name,self.target_dist))
#                 print(self.target_dist)
#                 if (self.target_dist > self.max_size_thresh):
#                     msg.linear.x = self.forward_chase_speed
#                     self.detect=True
                
#                 msg.angular.z = -self.angular_chase_multiplier*self.target_val
#                 # self.get_logger().info('Angula: {}'.format(msg.angular.z))
#             else:
#                 if self.pixel_y<180:
#                     # self.get_logger().info('Target lost')
#                     msg.angular.z = self.search_angular_speed
#         self.publisher_.publish(msg)

        
#     def deproject_pixel_to_point(self, K, pixel, depth):
#         fx = K[0, 0]
#         fy = K[1, 1]
#         cx = K[0, 2]
#         cy = K[1, 2]

#         x = (pixel[0] - cx) * depth / fx
#         y = (pixel[1] - cy) * depth / fy
#         z = float(depth)

#         return [x, y, z]

#     def publish_point(self, point_3d):
#         point_msg = PointStamped()
#         pose_msg = PoseStamped()
#         point_msg.header.stamp = self.get_clock().now().to_msg()
#         point_msg.header.frame_id = 'depth_camera'
#         point_msg.point.x = point_3d[0]
#         point_msg.point.y = point_3d[1]
#         point_msg.point.z = point_3d[2]
#         pose_msg.pose.position.x = abs(point_3d[0]/10)
#         pose_msg.pose.position.y = abs(point_3d[1]/10)
#         pose_msg.pose.position.z = abs(point_3d[2]/10)
#         quat = quaternion_from_euler(0, 160, 0)
#         pose_msg.pose.orientation.x = quat[0]
#         pose_msg.pose.orientation.y = quat[1]
#         pose_msg.pose.orientation.z = quat[2]
#         pose_msg.pose.orientation.w = quat[3]
#         self.publisher_point.publish(point_msg)
#         # self.get_logger().info(f"Published Point: {point_msg}")
#         # self.get_logger().info(f"Published Pose: {pose_msg}")
#         return pose_msg

#     def pixel_to_3d(self, x, y, depth):
#         fx = self.camera_info.k[0]
#         fy = self.camera_info.k[4]
#         cx = self.camera_info.k[2]
#         cy = self.camera_info.k[5]
        
#         x3d = ((x  - cx) * depth / fx)* 0.001
#         y3d = ((y  - cy) * depth / fy)* 0.001
#         z3d = depth * 0.001 
        
#         return x3d, y3d, z3d

#     def publish_tf(self, x, y, z, frame_id):
#         t = TransformStamped()
#         t.header.stamp = self.get_clock().now().to_msg()
#         t.header.frame_id = 'camera_color_optical_frame'
#         t.child_frame_id = frame_id
#         t.transform.translation.x = x
#         t.transform.translation.y = y
#         t.transform.translation.z = z
#         # Set rotation - in this example, we're not determining orientation
#         t.transform.rotation.w = 1.0
#         self.tf_broadcaster.sendTransform(t)
#     def camera_callback(self, data):
#         if self.depth_image is None :
#             return
#         # msg = Twist()

#         img = bridge.imgmsg_to_cv2(data, "bgr8")
#         resized_image = cv2.resize(img, (640, 480))
        
#         # results, ids = self.face_recognition.process_frame(img, self.recognition_on, self.registration_data)
#         results = self.model(resized_image, conf=0.5,verbose=False)
#         self.yolov8_inference.header.frame_id = "inference"
#         self.yolov8_inference.header.stamp = camera_subscriber.get_clock().now().to_msg()

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 self.inference_result = InferenceResult()
#                 b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
#                 c = box.cls
#                 self.inference_result.class_name = self.model.names[int(c)]
#                 self.inference_result.top = int(b[0])
#                 self.inference_result.left = int(b[1])
#                 self.inference_result.bottom = int(b[2])
#                 self.inference_result.right = int(b[3])
#                 self.pixel_x = int((self.inference_result.top+self.inference_result.bottom)/2)
#                 self.pixel_y = int((self.inference_result.right+self.inference_result.left)/2)
#                 self.yolov8_inference.yolov8_inference.append(self.inference_result)


#                 if len(self.depth_image)!=0:
#                     if self.inference_result.class_name=="sports ball" or self.inference_result.class_name=="frisbee":
#                         self.target_dist = self.depth_image[self.pixel_y, self.pixel_x]
#                         # self.get_logger().info('x=: {}'.format(self.pixel_x))
#                         # self.get_logger().info('y=: {}'.format(self.pixel_y))
#                         # self.get_logger().info('Depth=: {}'.format(self.target_dist))
#                         self.target_val=(self.pixel_x -320)/320
#                         K = np.array(self.camera_info.k).reshape(3, 3)
#                         point_3d = self.deproject_pixel_to_point(K, [self.pixel_x/1000, self.pixel_y/1000], self.target_dist)
#                         # print(f"Point 3d= {point_3d}")

#                         point_position = self.publish_point(point_3d)
#                         self.get_logger().info('Position: x= {}, y= {}, z= {}'.format(point_position.pose.position.x,point_position.pose.position.y,point_position.pose.position.z))
#                         self.get_logger().info('Orientation: x= {}, y= {}, z= {}'.format(point_position.pose.orientation.x,point_position.pose.orientation.y,point_position.pose.orientation.z))
#                         if self.target_dist>0:
#                             if self.target_dist<445 and self.detect:
#                                 self.findball=False
#                                 self.detect=False
#                                 self.get_logger().info("Send goal")
#                                 # self.get_logger().info('Position: x= {}, y= {}, z= {}'.format(point_position.pose.position.x,point_position.pose.position.y,point_position.pose.position.z))
#                                 self.send_goal(point_position.pose.position.x, point_position.pose.position.y,point_position.pose.position.z,point_position.pose.orientation.x,point_position.pose.orientation.y,point_position.pose.orientation.z,5)
#                                 # self.send_goal(point_3d[0]+0.28, point_3d[1],point_3d[2]+0.03,0.0,155.0,0.0,0)

                        
#                     else:

#                         self.target_dist=0
#                         self.pixel_y=0

#                     self.lastrcvtime = time.time()



#             # camera_subscriber.get_logger().info(f"{self.yolov8_inference}")

#         annotated_frame = results[0].plot()
#         img_msg = bridge.cv2_to_imgmsg(annotated_frame)  

#         self.img_pub.publish(img_msg)
#         self.yolov8_pub.publish(self.yolov8_inference)
#         self.yolov8_inference.yolov8_inference.clear()


#     def webcam_callback(self, data):
#         # msg = Twist()

#         img = bridge.imgmsg_to_cv2(data, "bgr8")
        
#         results, ids = self.face_recognition.process_frame(img, self.recognition_on, self.registration_data)
#         results = self.model(results, conf=0.5)
#         self.yolov8_inference.header.frame_id = "inference"
#         self.yolov8_inference.header.stamp = camera_subscriber.get_clock().now().to_msg()

#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 self.inference_result = InferenceResult()
#                 b = box.xyxy[0].to('cpu').detach().numpy().copy()  # get box coordinates in (top, left, bottom, right) format
#                 c = box.cls
#                 self.inference_result.class_name = self.model.names[int(c)]
#                 self.inference_result.top = int(b[0])
#                 self.inference_result.left = int(b[1])
#                 self.inference_result.bottom = int(b[2])
#                 self.inference_result.right = int(b[3])
#                 self.pixel_x = int((self.inference_result.top+self.inference_result.bottom)/2)
#                 self.pixel_y = int((self.inference_result.right+self.inference_result.left)/2)
#                 self.yolov8_inference.yolov8_inference.append(self.inference_result)


#                 # if len(self.depth_image)!=0:
#                 #     if self.inference_result.class_name=="sports ball":
#                 #         self.target_dist = self.depth_image[self.pixel_y, self.pixel_x]
#                 #         self.get_logger().info('x=: {}'.format(self.pixel_x))
#                 #         self.get_logger().info('y=: {}'.format(self.pixel_y))
#                 #         self.target_val=(self.pixel_x -320)/320


                        
#                 #     else:

#                 #         self.target_dist=0
#                 #         self.pixel_y=0

#                 #     self.lastrcvtime = time.time()
#                 #     if self.target_dist<0.1 and self.detect:
#                 #         self.send_goal(0.13987954042851925, -2.1001233108108863e-07,0.11195994687080383,0.0,180.0,0.0,0)


#             camera_subscriber.get_logger().info(f"{self.yolov8_inference}")

#         annotated_frame = results[0].plot()
#         img_msg = bridge.cv2_to_imgmsg(annotated_frame)  

#         self.img_pub.publish(img_msg)
#         self.yolov8_pub.publish(self.yolov8_inference)
#         self.yolov8_inference.yolov8_inference.clear()


#     def send_goal(self, p_x,p_y,p_z,or_x,or_y,or_z,order):
#         self.detect=False
#         goal_msg = ArticubotTask.Goal()
#         goal_msg.task = order
#         goal_msg.p_x=p_x
#         goal_msg.p_y=p_y
#         goal_msg.p_z=p_z
#         goal_msg.or_x=or_x
#         goal_msg.or_y=or_y
#         goal_msg.or_z=or_z
#         self._action_client.wait_for_server()
#         self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
#         self.get_logger().info('Result: {0}'.format(self._send_goal_future.add_done_callback(self.goal_response_callback)))
#         self._send_goal_future.add_done_callback(self.goal_response_callback)
        
#     def get_result_callback(self, future):
#         result = future.result().result
#         status = future.result().status
#         self.get_logger().info('Result: {0}'.format(result.success))
#         self.get_logger().info('Status: {0}'.format(status))
#         self.detect=True
        



#     def feedback_callback(self, feedback_msg):
#             feedback = feedback_msg.feedback
#             self.get_logger().info('Received feedback:',feedback)

#     def goal_response_callback(self, future):
#         goal_handle = future.result()
#         if not goal_handle.accepted:
#             self.get_logger().info('Goal rejected :(')
#             return
#         self.get_logger().info('Goal accepted :)')
#         self._get_result_future = goal_handle.get_result_async()
#         self._get_result_future.add_done_callback(self.get_result_callback)   

#     def depth_camera_callback(self,data):
#         self.depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
#         self.depth_value = self.depth_image[self.pixel_y, self.pixel_x]
#         # self.get_logger().info(f"Depth value at depth {self.depth_value}")

#     def findball_callback(self, msg):
#         if "true" in msg.data.lower():
#             self.findball = True
#         else :
#             self.findball = False

# if __name__ == '__main__':
#     rclpy.init(args=None)
#     camera_subscriber = Camera_subscriber()
#     rclpy.spin(camera_subscriber)
#     rclpy.shutdown()






#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from tf_transformations import quaternion_from_euler
from articubot_msgs.msg import InferenceResult, Yolov8Inference
from articubot_msgs.action import ArticubotTask
from rclpy.action import ActionClient
import time
from std_msgs.msg import String
import psutil

class Camera_subscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')
        self.get_logger().info("Camera subscriber node initialized")

        # Load YOLOv8 model
        self.model = YOLO('yolov8m.pt')  # Use the nano model
        self.model.fuse()  # Fuse layers for inference
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
            self.get_logger().info("CUDA is available. Using GPU for inference.")
        else:
            self.get_logger().info("CUDA is not available. Using CPU for inference.")

        # Publishers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 5)
        self.img_pub = self.create_publisher(Image, "/inference_result", 5)
        self.publisher_point = self.create_publisher(PointStamped, 'point_3d', 10)

        # Subscribers
        self.subscription = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.camera_callback,
            10)
        self.subscription_depth = self.create_subscription(
            Image,
            'camera/camera/depth/image_rect_raw',
            self.depth_camera_callback,
            10)
        self.create_subscription(
            CameraInfo,
            'camera/camera/depth/camera_info',
            self.camera_info_callback,
            10)
        self.sub_supress = self.create_subscription(
            String, 'find_ball', self.findball_callback, 10)

        # Action client
        self._action_client = ActionClient(self, ArticubotTask, 'test_server')

        # Other initializations
        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_image = None
        self.last_process_time = self.get_clock().now().to_msg().sec
        self.process_interval = 1.0  # Increase to 1 second
        self.findball = True
        self.target_dist = 0.0
        self.target_val = 0.0
        self.pixel_x = 0
        self.pixel_y = 0
        self.detect = False
        self.previous_frame = None
        self.motion_threshold = 25  # Adjust as needed

        # Create a timer for periodic processing
        self.create_timer(0.1, self.timer_callback)

        # Resource monitoring
        self.create_timer(5.0, self.monitor_resources)

        self.get_logger().info("Camera subscriber node setup complete")

    def camera_callback(self, msg):
        current_time = self.get_clock().now().to_msg().sec
        if current_time - self.last_process_time < self.process_interval:
            return  # Skip this frame

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # More aggressive downsampling
            resized_image = cv2.resize(cv_image, (320, 240))  # Reduced from 640x480

            # Motion detection
            if not self.detect_motion(resized_image):
                return  # Skip processing if no significant motion

            self.last_process_time = current_time

            with torch.no_grad():  # Disable gradient calculation
                results = self.model(resized_image, conf=0.5, verbose=False)
            self.process_results(results, cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def detect_motion(self, frame):
        if self.previous_frame is None:
            self.previous_frame = frame
            return True

        frame_diff = cv2.absdiff(frame, self.previous_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
        motion_detected = np.sum(motion_mask) > self.motion_threshold

        self.previous_frame = frame
        return motion_detected

    def process_results(self, results, original_image):
        yolov8_inference = Yolov8Inference()
        yolov8_inference.header.frame_id = "inference"
        yolov8_inference.header.stamp = self.get_clock().now().to_msg()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                inference_result = InferenceResult()
                b = box.xyxy[0].cpu().numpy()
                c = box.cls
                inference_result.class_name = self.model.names[int(c)]
                inference_result.top = int(b[0])
                inference_result.left = int(b[1])
                inference_result.bottom = int(b[2])
                inference_result.right = int(b[3])
                
                self.pixel_x = int((inference_result.top + inference_result.bottom) / 2)
                self.pixel_y = int((inference_result.right + inference_result.left) / 2)
                
                yolov8_inference.yolov8_inference.append(inference_result)

                if self.depth_image is not None and (inference_result.class_name == "sports ball" or inference_result.class_name == "frisbee"):
                    self.process_depth(inference_result)

        # Only publish annotated frame if objects were detected
        if len(yolov8_inference.yolov8_inference) > 0:
            annotated_frame = results[0].plot()
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.img_pub.publish(img_msg)
            self.yolov8_pub.publish(yolov8_inference)

    def process_depth(self, inference_result):
        self.target_dist = self.depth_image[self.pixel_y, self.pixel_x]
        self.target_val = (self.pixel_x - 160) / 160  # Adjusted for 320x240 resolution

        if self.camera_info:
            K = np.array(self.camera_info.k).reshape(3, 3)
            point_3d = self.deproject_pixel_to_point(K, [self.pixel_x/500, self.pixel_y/500], self.target_dist)
            point_position = self.publish_point(point_3d)

            if 0 < self.target_dist < 445 and self.detect:
                self.findball = False
                self.detect = False
                self.get_logger().info("Sending goal for object interaction")
                self.send_goal(point_position.pose.position.x, 
                               point_position.pose.position.y,
                               point_position.pose.position.z,
                               point_position.pose.orientation.x,
                               point_position.pose.orientation.y,
                               point_position.pose.orientation.z,
                               5)

    def timer_callback(self):
        if not self.findball:
            return

        msg = Twist()
        if self.target_dist > 0.1:  # Adjust this threshold as needed
            msg.linear.x = 0.1  # Adjust speed as needed
            msg.angular.z = -0.2 * self.target_val  # Adjust turning rate as needed
        else:
            msg.angular.z = 0.15  # Search rotation speed

        self.publisher_.publish(msg)

    def depth_camera_callback(self, data):
        self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def findball_callback(self, msg):
        self.findball = "true" in msg.data.lower()

    def deproject_pixel_to_point(self, K, pixel, depth):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x = (pixel[0] - cx) * depth / fx
        y = (pixel[1] - cy) * depth / fy
        return [x, y, depth]

    def publish_point(self, point_3d):
        point_msg = PointStamped()
        pose_msg = PoseStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = 'depth_camera'
        point_msg.point.x, point_msg.point.y, point_msg.point.z = point_3d

        pose_msg.pose.position.x = abs(point_3d[0]/10)
        pose_msg.pose.position.y = abs(point_3d[1]/10)
        pose_msg.pose.position.z = abs(point_3d[2]/10)
        quat = quaternion_from_euler(0, 160, 0)
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat

        self.publisher_point.publish(point_msg)
        return pose_msg

    def send_goal(self, p_x, p_y, p_z, or_x, or_y, or_z, order):
        goal_msg = ArticubotTask.Goal()
        goal_msg.task = order
        goal_msg.p_x, goal_msg.p_y, goal_msg.p_z = p_x, p_y, p_z
        goal_msg.or_x, goal_msg.or_y, goal_msg.or_z = or_x, or_y, or_z
        
        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Received feedback: {feedback_msg.feedback}')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}')
        self.detect = True

    def monitor_resources(self):
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        self.get_logger().info(f"CPU usage: {cpu_percent}%, Memory usage: {memory_percent}%")

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = Camera_subscriber()
    try:
        rclpy.spin(camera_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        camera_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()