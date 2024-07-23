


// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// #include <tf2_ros/transform_listener.h>
// #include <geometry_msgs/msg/transform_stamped.hpp>

// #include <memory>

// #include <rclcpp/rclcpp.hpp>
// #include <moveit/move_group_interface/move_group_interface.h>
// #include <rclcpp_action/rclcpp_action.hpp>


//   double to_radians(const double deg_angle)
// {
//   return deg_angle * M_PI / 180.0;
// }
// int main(int argc, char* argv[])
// {
//   // Initialize ROS and create the Node
//   rclcpp::init(argc, argv);

//   auto const node = std::make_shared<rclcpp::Node>(
//       "hello_moveit", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

//   // Create a ROS logger
//   auto const logger = rclcpp::get_logger("hello_moveit");
  

//   // Create the MoveIt MoveGroup Interface
//   using moveit::planning_interface::MoveGroupInterface;
//   auto move_group_interface = MoveGroupInterface(node, "arm_robot");

//   // Set a target Pose
//   auto const target_pose = [] {
//     geometry_msgs::msg::Pose target_pose;
//     tf2::Quaternion q;
//     target_pose.position.x = 0.2747167457640171;  // Example values, adjust according to your setup
//     target_pose.position.y = -2.249231717017409e-10;
//     target_pose.position.z = 0.150327756881713867;
//     q.setRPY(to_radians(0), to_radians(155), to_radians(0));
//     target_pose.orientation = tf2::toMsg(q);
//     return target_pose;
//   }();
//   move_group_interface.setPoseTarget(target_pose);

//   // Create a plan to that target pose
//   auto const [success, plan] = [&move_group_interface] {
//     moveit::planning_interface::MoveGroupInterface::Plan msg;
//     auto const ok = static_cast<bool>(move_group_interface.plan(msg));
//     return std::make_pair(ok, msg);
//   }();

//   // Execute the plan
//   if (success)
//   {
//     move_group_interface.execute(plan);
//   geometry_msgs::msg::Pose current_pose =
//     move_group_interface.getCurrentPose().pose;

//   // Print the current pose of the end effector
//   RCLCPP_INFO(node->get_logger(), "Current pose: %f %f %f %f %f %f %f",
//     current_pose.position.x,
//     current_pose.position.y,
//     current_pose.position.z,
//     current_pose.orientation.x,
//     current_pose.orientation.y,
//     current_pose.orientation.z,
//     current_pose.orientation.w);
//   }
//   else
//   {
//     RCLCPP_ERROR(logger, "Planing failed!");
//   }

//   // Shutdown ROS
//   rclcpp::shutdown();
//   return 0;
// }

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/utils.h"

using namespace std::chrono_literals;

class LinkPoseGetter : public rclcpp::Node
{
public:
  LinkPoseGetter()
  : Node("link_pose_getter")
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    timer_ = this->create_wall_timer(
      1000ms, std::bind(&LinkPoseGetter::timer_callback, this));

    // Replace 'your_link_name' with the name of the link you want to track
    link_name_ = "claw_support";
  }

private:
  void timer_callback()
  {
    geometry_msgs::msg::TransformStamped transformStamped;

    try {
      transformStamped = tf_buffer_->lookupTransform(
        "base_link", link_name_, tf2::TimePointZero);
    } catch (tf2::TransformException & ex) {
      RCLCPP_INFO(
        this->get_logger(), "Could not transform %s to world: %s",
        link_name_.c_str(), ex.what());
      return;
    }

    auto translation = transformStamped.transform.translation;
    auto rotation = transformStamped.transform.rotation;

    // Convert quaternion to roll, pitch, yaw
    double roll, pitch, yaw;
    tf2::Quaternion q(
      rotation.x,
      rotation.y,
      rotation.z,
      rotation.w);
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    RCLCPP_INFO(
      this->get_logger(),
      "Pose of %s:\n"
      "Position: x=%.2f, y=%.2f, z=%.2f\n"
      "Orientation: x=%.2f, y=%.2f, z=%.2f, w=%.2f\n"
      "RPY (radians): roll=%.2f, pitch=%.2f, yaw=%.2f\n"
      "RPY (degrees): roll=%.2f, pitch=%.2f, yaw=%.2f",
      link_name_.c_str(),
      translation.x, translation.y, translation.z,
      rotation.x, rotation.y, rotation.z, rotation.w,
      roll, pitch, yaw,
      roll * 180.0 / M_PI, pitch * 180.0 / M_PI, yaw * 180.0 / M_PI);
  }

  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::string link_name_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LinkPoseGetter>());
  rclcpp::shutdown();
  return 0;
}