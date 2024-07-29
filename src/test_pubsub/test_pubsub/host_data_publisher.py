import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil
import json
import socket

class HostDataPublisher(Node):

    def __init__(self):
        super().__init__('host_data_publisher')
        self.publisher_ = self.create_publisher(String, 'host_data', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def get_host_data(self):
        data = {
            "hostname": socket.gethostname(),
            "ip": socket.gethostbyname(socket.gethostname()),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
        return json.dumps(data)

    def timer_callback(self):
        msg = String()
        msg.data = self.get_host_data()
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    host_data_publisher = HostDataPublisher()
    rclpy.spin(host_data_publisher)
    host_data_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()