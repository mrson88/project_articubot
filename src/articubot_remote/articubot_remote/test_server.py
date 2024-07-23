#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
import math
from tf_transformations import euler_from_quaternion

class LinkPoseGetter(Node):

    def __init__(self):
        super().__init__('link_pose_getter')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create a timer that will trigger the callback every 1 second
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Replace 'your_link_name' with the name of the link you want to track
        self.link_name = 'claw_support'

    def timer_callback(self):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'base_link',
                self.link_name,
                rclpy.time.Time())

            # Extract the position
            position = transform.transform.translation

            # Extract the orientation
            orientation = transform.transform.rotation

            # Convert quaternion to Euler angles (roll, pitch, yaw)
            quaternion = [
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            ]
            roll, pitch, yaw = euler_from_quaternion(quaternion)

            self.get_logger().info(
                f'Pose of {self.link_name}:\n'
                f'Position: x={position.x:.2f}, y={position.y:.2f}, z={position.z:.2f}\n'
                f'Orientation (quaternion): x={orientation.x:.2f}, y={orientation.y:.2f}, '
                f'z={orientation.z:.2f}, w={orientation.w:.2f}\n'
                f'Orientation (RPY radians): roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}\n'
                f'Orientation (RPY degrees): roll={math.degrees(roll):.2f}, '
                f'pitch={math.degrees(pitch):.2f}, yaw={math.degrees(yaw):.2f}'
            )

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {self.link_name} to world: {ex}')

def main():
    rclpy.init()
    node = LinkPoseGetter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()