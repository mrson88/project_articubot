#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include "articubot_msgs/action/articubot_task.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include "geometry_msgs/msg/quaternion.hpp"
#include <memory>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include <string>
  double to_radians(const double deg_angle)
{
  return deg_angle * M_PI / 180.0;
}

using namespace std::placeholders;

namespace articubot_remote
{
class Test_server : public rclcpp::Node
{
public:
  explicit Test_server(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("test_server", options)
  {
    RCLCPP_INFO(get_logger(), "Starting the Test Server");
    action_server_ = rclcpp_action::create_server<articubot_msgs::action::ArticubotTask>(
        this, "test_server", std::bind(&Test_server::goalCallback, this, _1, _2),
        std::bind(&Test_server::cancelCallback, this, _1),
        std::bind(&Test_server::acceptedCallback, this, _1));
  }

private:
  rclcpp_action::Server<articubot_msgs::action::ArticubotTask>::SharedPtr action_server_;

  rclcpp_action::GoalResponse goalCallback(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const articubot_msgs::action::ArticubotTask::Goal> goal)
  {
    RCLCPP_INFO(get_logger(), "Received goal request with id %d", goal->task);
    (void)uuid;
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse cancelCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    auto arm_move_group = moveit::planning_interface::MoveGroupInterface(shared_from_this(), "arm_robot");
    auto gripper_move_group = moveit::planning_interface::MoveGroupInterface(shared_from_this(), "gripper");
    arm_move_group.stop();
    gripper_move_group.stop();
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void acceptedCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
  {
    // this needs to return quickly to avoid blocking the executor, so spin up a new thread
    std::thread{ std::bind(&Test_server::execute, this, _1), goal_handle }.detach();
  }

  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)



  {
    RCLCPP_INFO(get_logger(), "Executing goal");
    auto result = std::make_shared<articubot_msgs::action::ArticubotTask::Result>();

    // MoveIt 2 Interface
  auto const node = std::make_shared<rclcpp::Node>(
  "test_server_moveit_robot", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // Create a ROS logger
  auto const logger = rclcpp::get_logger("test_server_moveit_robot");

  // Create the MoveIt MoveGroup Interface
  using moveit::planning_interface::MoveGroupInterface;
  auto move_group_interface = MoveGroupInterface(node, "arm_robot");
   auto move_group_gripper_interface = MoveGroupInterface(node, "gripper");
  std::vector<double> gripper_joint_values;

  double GRIPPER_DEFAULT = to_radians(-20);
  double GRIPPER_OPEN = to_radians(-40);
  double GRIPPER_CLOSE = to_radians(10);

  double goal_p_x = goal_handle->get_goal()->p_x;
  double goal_p_y = goal_handle->get_goal()->p_y;
  double goal_p_z = goal_handle->get_goal()->p_z;
  double goal_or_x = goal_handle->get_goal()->or_x;
  double goal_or_y = goal_handle->get_goal()->or_y;
  double goal_or_z = goal_handle->get_goal()->or_z;
  double goal_or_w = goal_handle->get_goal()->or_w;

  // const moveit::core::JointModelGroup* joint_model_group =
  //     move_group_interface.getCurrentState()->getJointModelGroup("arm_robot");

    if (goal_handle->get_goal()->task == 0)
    {
      std::vector<double> gripper_joint_goal;
      geometry_msgs::msg::Pose msg;
      tf2::Quaternion q;
      gripper_joint_goal = {GRIPPER_OPEN};
      move_group_gripper_interface.setJointValueTarget(gripper_joint_goal);
      move_group_gripper_interface.move();
      msg.position.x = goal_p_x;
      msg.position.y = goal_p_y;
      msg.position.z = goal_p_z;
      q.setRPY(to_radians(goal_or_x), to_radians(goal_or_y), to_radians(goal_or_z));
      msg.orientation = tf2::toMsg(q);
      move_group_interface.setPoseTarget(msg);

      RCLCPP_INFO(get_logger(), "Executing goal 0");  
      move_group_interface.move();


      gripper_joint_values = {GRIPPER_CLOSE};
      move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
      move_group_gripper_interface.move();

      // move_group_gripper_interface.move();
      // move_group_interface.setNamedTarget("vertical");
      // move_group_interface.move();
      move_group_interface.setNamedTarget("home");
      move_group_interface.move();
      gripper_joint_values = {GRIPPER_DEFAULT};
      move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
      move_group_gripper_interface.move();


      move_group_interface.setNamedTarget("vertical");
      move_group_interface.move();
      std::vector<double> arm_joint_goal;
      arm_joint_goal = {0.0, -0.5, 1.0,1.2, 0.5};

      move_group_interface.setJointValueTarget(arm_joint_goal);
      move_group_interface.move();
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(get_logger(), "Goal succeeded");
      
    }
    else if (goal_handle->get_goal()->task == 1)
    {
      geometry_msgs::msg::Pose msg;
      msg.position.x = goal_p_x;
      msg.position.y = goal_p_y;
      msg.position.z = goal_p_z;
      msg.orientation.x = goal_or_x;
      msg.orientation.y = goal_or_y;
      msg.orientation.z = goal_or_z;
      msg.orientation.w = goal_or_w;

      move_group_interface.setPoseTarget(msg);
      std::vector<double> gripper_joint_goal;

      RCLCPP_INFO(get_logger(), "Executing goal 1");  
      move_group_interface.move();
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(get_logger(), "Goal succeeded");
    }
    else if (goal_handle->get_goal()->task == 2)
    {

      moveit::core::RobotStatePtr current_state = move_group_interface.getCurrentState(10);
      std::vector<double> joint_group_positions;
      // current_state->copyJointGroupPositions(joint_model_group, joint_group_positions);
      joint_group_positions[0] = -1.0;  // radians
      // move_group_interface.setJointValueTarget(joint_group_positions);
      // std::vector<double> arm_joint_goal;
      // arm_joint_goal = {0.5, -1.2, -1.2,-1.2, 0.5};

      // move_group_interface.setJointValueTarget(arm_joint_goal);
      // move_group_interface.setNamedTarget("home");
      // moveit::planning_interface::MoveGroupInterface::Plan arm_plan;
      // bool arm_plan_success = (arm_move_group.plan(arm_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      move_group_interface.move();
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Invalid Task Number");
      return;
    }


  // // Create a plan to that target pose
  // auto const [success, plan] = [&move_group_interface] {
  //   moveit::planning_interface::MoveGroupInterface::Plan msg;
  //   auto const ok = static_cast<bool>(move_group_interface.plan(msg));
  //   return std::make_pair(ok, msg);
  // }();

  // // Execute the plan
  // if (success)
  // {
  //   // move_group_interface.execute(plan);
  //   // move_group_interface.move();
  //   // move_group_interface.setNamedTarget("home");
  //   move_group_interface.move();

  // }
  // else
  // {
  //   RCLCPP_ERROR(logger, "Planing failed!");
  //   return;
  // }
  //     result->success = true;
  //   goal_handle->succeed(result);
  //   RCLCPP_INFO(get_logger(), "Goal succeeded");
  }

};
}  // namespace arduinobot_remote

RCLCPP_COMPONENTS_REGISTER_NODE(articubot_remote::Test_server)