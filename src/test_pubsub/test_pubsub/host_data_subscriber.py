import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json

class HostDataSubscriber(Node):

    def __init__(self):
        super().__init__('host_data_subscriber')
        self.subscription = self.create_subscription(
            String,
            'host_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        data = json.loads(msg.data)
        self.get_logger().info('Received host data:')
        self.get_logger().info(f"Hostname: {data['hostname']}")
        self.get_logger().info(f"IP: {data['ip']}")
        self.get_logger().info(f"CPU Usage: {data['cpu_percent']}%")
        self.get_logger().info(f"Memory Usage: {data['memory_percent']}%")
        self.get_logger().info(f"Disk Usage: {data['disk_usage']}%")

def main(args=None):
    rclpy.init(args=args)
    host_data_subscriber = HostDataSubscriber()
    rclpy.spin(host_data_subscriber)
    host_data_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()