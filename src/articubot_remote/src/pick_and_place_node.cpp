


#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp_action/rclcpp_action.hpp>


  double to_radians(const double deg_angle)
{
  return deg_angle * M_PI / 180.0;
}
int main(int argc, char* argv[])
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);

  auto const node = std::make_shared<rclcpp::Node>(
      "pick_and_place", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // Create a ROS logger
  auto const logger = rclcpp::get_logger("pick_and_place");
  

  // Create the MoveIt MoveGroup Interface
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "arm_robot");

  // Set a target Pose
  auto const target_pose = [] {
    geometry_msgs::msg::Pose target_pose;

    tf2::Quaternion q;
    target_pose.position.x = 0.2747167457640171;  // Example values, adjust according to your setup
    target_pose.position.y = -2.249231717017409e-10;
    target_pose.position.z = 0.150327756881713867;
    q.setRPY(to_radians(0), to_radians(155), to_radians(0));
    target_pose.orientation = tf2::toMsg(q);
    return target_pose;
  }();
  move_group_interface.setPoseTarget(target_pose);

  // Create a plan to that target pose
  auto const [success, plan] = [&move_group_interface] {
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

  // Execute the plan
  if (success)
  {
    move_group_interface.execute(plan);
  // geometry_msgs::msg::Pose current_pose =
  //   move_group_interface.getCurrentPose().pose;
  //   geometry_msgs::msg::PoseStamped current_pose = move_group_interface.getCurrentPose();

  // // Print the current pose of the end effector
  // RCLCPP_INFO(node->get_logger(), "Current pose: %f %f %f %f %f %f %f",
  //   current_pose.pose.position.x,
  //   current_pose.pose.position.y,
  //   current_pose.pose.position.z,
  //   current_pose.pose.orientation.x,
  //   current_pose.pose.orientation.y,
  //   current_pose.pose.orientation.z,
  //   current_pose.pose.orientation.w);

  moveit::planning_interface::MoveGroupInterface move_group(node,"arm_robot");

  geometry_msgs::msg::PoseStamped current_pose = move_group.getCurrentPose();

  RCLCPP_INFO(node->get_logger(),"Current end-effector pose: " , current_pose);
  }
  else
  {
    RCLCPP_ERROR(logger, "Planing failed!");
  }

  // Shutdown ROS
  rclcpp::shutdown();
  return 0;
}