#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include "articubot_msgs/action/articubot_task.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include "geometry_msgs/msg/quaternion.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <memory>
#include <string>
#include <thread>

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
        this, "test_server", 
        std::bind(&Test_server::goalCallback, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&Test_server::cancelCallback, this, std::placeholders::_1),
        std::bind(&Test_server::acceptedCallback, this, std::placeholders::_1));
    subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&Test_server::joint_states_callback, this, std::placeholders::_1));
  }

private:
  rclcpp_action::Server<articubot_msgs::action::ArticubotTask>::SharedPtr action_server_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  std::vector<double> current_joint = {0.0, -0.5, 1.0, 1.2, 0.5, 0, 0};

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
    RCLCPP_INFO(get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void acceptedCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
  {
    std::thread{std::bind(&Test_server::execute, this, std::placeholders::_1), goal_handle}.detach();
  }

  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->position.size(); i++)
    {
      current_joint[i] = msg->position[i];
    }
  }

  void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
  {
    RCLCPP_INFO(get_logger(), "Executing goal");
    auto result = std::make_shared<articubot_msgs::action::ArticubotTask::Result>();

    auto move_group_node = rclcpp::Node::make_shared("move_group_interface_tutorial");
    
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(move_group_node);
    std::thread([&executor]() { executor.spin(); }).detach();

    static const std::string PLANNING_GROUP = "arm_robot";
    auto move_group_interface = std::make_shared<moveit::planning_interface::MoveGroupInterface>(move_group_node, PLANNING_GROUP);
    auto move_group_gripper_interface = std::make_shared<moveit::planning_interface::MoveGroupInterface>(move_group_node, "gripper");

    setupMoveGroupInterface(move_group_interface);

    bool success = false;
    switch (goal_handle->get_goal()->task) {
      case 0:
        success = moveToPoint(move_group_interface, move_group_gripper_interface, goal_handle);
        break;
      case 1:
        success = moveToHome(move_group_interface);
        break;
      case 2:
        success = closeGripper(move_group_gripper_interface);
        break;
      default:
        RCLCPP_ERROR(get_logger(), "Invalid Task Number");
        result->success = false;
        goal_handle->abort(result);
        return;
    }

    if (success) {
      logRobotState(move_group_interface);
      result->success = true;
      goal_handle->succeed(result);
    } else {
      RCLCPP_ERROR(get_logger(), "Task execution failed");
      result->success = false;
      goal_handle->abort(result);
    }
  }

  void setupMoveGroupInterface(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
  {
    move_group_interface->setPlanningTime(10.0);
    move_group_interface->setNumPlanningAttempts(10);
    move_group_interface->setMaxVelocityScalingFactor(0.1);
    move_group_interface->setMaxAccelerationScalingFactor(0.1);
  }

  bool moveToPoint(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                   const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
                   const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>>& goal_handle)
  {
    geometry_msgs::msg::Pose target_pose = createTargetPose(goal_handle);
    return planAndExecute(move_group_interface, move_group_gripper_interface, target_pose);
  }

  bool moveToHome(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
  {
    RCLCPP_INFO(get_logger(), "Moving to home position...");
    move_group_interface->setNamedTarget("home");
    return (move_group_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool closeGripper(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface)
  {
    RCLCPP_INFO(get_logger(), "Closing gripper...");
    move_group_gripper_interface->setNamedTarget("closed");
    return (move_group_gripper_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

  geometry_msgs::msg::Pose createTargetPose(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>>& goal_handle)
  {
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = goal_handle->get_goal()->p_x;
    target_pose.position.y = goal_handle->get_goal()->p_y;
    target_pose.position.z = goal_handle->get_goal()->p_z;
    target_pose.orientation.x = goal_handle->get_goal()->or_x;
    target_pose.orientation.y = goal_handle->get_goal()->or_y;
    target_pose.orientation.z = goal_handle->get_goal()->or_z;
    target_pose.orientation.w = goal_handle->get_goal()->or_w;
    return target_pose;
  }

  bool planAndExecute(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                      const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
                      const geometry_msgs::msg::Pose& target_pose)
  {
    move_group_interface->setPoseTarget(target_pose);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = false;

    std::vector<std::string> planners = {"RRTConnect", "RRTstar", "PRMstar", "LBKPIECE", "BKPIECE"};
    
    for (const auto& planner : planners) {
      move_group_interface->setPlannerId(planner);
      RCLCPP_INFO(get_logger(), "Attempting to plan with %s", planner.c_str());
      success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      if (success) {
        RCLCPP_INFO(get_logger(), "Planning succeeded with %s", planner.c_str());
        break;
      }
    }

    if (!success) {
      RCLCPP_ERROR(get_logger(), "Planning failed with all planners");
      success = attemptCartesianPath(move_group_interface, target_pose);
    }

    if (success) {
      RCLCPP_INFO(get_logger(), "Execution starting...");
      move_group_gripper_interface->setNamedTarget("open");
      move_group_gripper_interface->move();
      success = (move_group_interface->execute(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      if (!success) {
        RCLCPP_ERROR(get_logger(), "Execution failed");
      }
    }

    return success;
  }

  bool attemptCartesianPath(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                            const geometry_msgs::msg::Pose& target_pose)
  {
    RCLCPP_INFO(get_logger(), "Attempting Cartesian path planning...");
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(move_group_interface->getCurrentPose().pose);
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_interface->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

    if (fraction > 0.0) {
      RCLCPP_INFO(get_logger(), "Cartesian path (%.2f%% achieved)", fraction * 100.0);
      return (move_group_interface->execute(trajectory) == moveit::core::MoveItErrorCode::SUCCESS);
    } else {
      RCLCPP_ERROR(get_logger(), "Cartesian path planning failed");
      logPlanningFailure(move_group_interface, target_pose);
      return false;
    }
  }

  void logPlanningFailure(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                          const geometry_msgs::msg::Pose& target_pose)
  {
    auto current_pose = move_group_interface->getCurrentPose().pose;
    RCLCPP_INFO(get_logger(), "Current position: x=%.3f, y=%.3f, z=%.3f", 
                current_pose.position.x, current_pose.position.y, current_pose.position.z);
    RCLCPP_INFO(get_logger(), "Target position: x=%.3f, y=%.3f, z=%.3f", 
                target_pose.position.x, target_pose.position.y, target_pose.position.z);

    const moveit::core::JointModelGroup* joint_model_group = 
      move_group_interface->getCurrentState()->getJointModelGroup(move_group_interface->getName());
    
    for (const auto& joint_name : joint_model_group->getVariableNames()) {
      const moveit::core::VariableBounds& bounds = move_group_interface->getCurrentState()->getJointModel(joint_name)->getVariableBounds()[0];
      if (bounds.position_bounded_) {
        RCLCPP_INFO(get_logger(), "Joint %s bounds: [%.3f, %.3f]", 
                    joint_name.c_str(), bounds.min_position_, bounds.max_position_);
      }
    }
  }

  void logRobotState(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
  {
    RCLCPP_INFO(get_logger(), "Planning frame: %s", move_group_interface->getPlanningFrame().c_str());
    RCLCPP_INFO(get_logger(), "End effector link: %s", move_group_interface->getEndEffectorLink().c_str());
    RCLCPP_INFO(get_logger(), "Available planning groups:");
    std::vector<std::string> group_names = move_group_interface->getJointModelGroupNames();
    for (const auto& group_name : group_names) {
      RCLCPP_INFO(get_logger(), "  %s", group_name.c_str());
    }

    for (size_t i = 0; i < current_joint.size(); i++) {
      RCLCPP_INFO(get_logger(), "Joint %ld: %f", i, current_joint[i]);
    }

    auto current_pose = move_group_interface->getCurrentPose().pose;
    RCLCPP_INFO(get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
                current_pose.position.x, current_pose.position.y, current_pose.position.z,
                current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
  }
};

}  // namespace articubot_remote

RCLCPP_COMPONENTS_REGISTER_NODE(articubot_remote::Test_server)













// #include <tf2/LinearMath/Quaternion.h>
// #include <tf2/LinearMath/Matrix3x3.h>
// #include <rclcpp/rclcpp.hpp>
// #include <rclcpp_action/rclcpp_action.hpp>
// #include <rclcpp_components/register_node_macro.hpp>
// #include "articubot_msgs/action/articubot_task.hpp"
// #include <moveit/move_group_interface/move_group_interface.h>
// #include <moveit/planning_scene_interface/planning_scene_interface.h>
// #include "geometry_msgs/msg/quaternion.hpp"
// #include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// #include <moveit/robot_state/robot_state.h>
// #include <moveit/robot_model/robot_model.h>
// #include <moveit/planning_scene/planning_scene.h>
// #include <moveit_msgs/msg/display_trajectory.hpp>
// #include <moveit_msgs/msg/planning_scene.hpp>
// #include <memory>
// #include <string>
// #include <thread>

// double to_radians(const double deg_angle)
// {
//   return deg_angle * M_PI / 180.0;
// }

// std::vector<double> current_joint = {0.0, -0.5, 1.0, 1.2, 0.5, 0, 0};

// using namespace std::placeholders;

// namespace articubot_remote
// {

// class Test_server : public rclcpp::Node
// {
// public:
//   explicit Test_server(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
//     : Node("test_server", options)
//   {
//     RCLCPP_INFO(get_logger(), "Starting the Test Server");
//     action_server_ = rclcpp_action::create_server<articubot_msgs::action::ArticubotTask>(
//         this, "test_server", std::bind(&Test_server::goalCallback, this, _1, _2),
//         std::bind(&Test_server::cancelCallback, this, _1),
//         std::bind(&Test_server::acceptedCallback, this, _1));
//     subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
//         "/joint_states", 10, std::bind(&Test_server::joint_states_callback, this, std::placeholders::_1));
//   }

// private:
//   rclcpp_action::Server<articubot_msgs::action::ArticubotTask>::SharedPtr action_server_;
//   rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;

//   rclcpp_action::GoalResponse goalCallback(
//       const rclcpp_action::GoalUUID& uuid,
//       std::shared_ptr<const articubot_msgs::action::ArticubotTask::Goal> goal)
//   {
//     RCLCPP_INFO(get_logger(), "Received goal request with id %d", goal->task);
//     (void)uuid;
//     return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
//   }

//   rclcpp_action::CancelResponse cancelCallback(
//       const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
//   {
//     RCLCPP_INFO(get_logger(), "Received request to cancel goal");
//     (void)goal_handle;
//     return rclcpp_action::CancelResponse::ACCEPT;
//   }

//   void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
//   {
//     for (size_t i = 0; i < msg->position.size(); i++)
//     {
//       current_joint[i] = msg->position[i];
//     }
//   }

//   void acceptedCallback(
//       const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
//   {
//     std::thread{std::bind(&Test_server::execute, this, _1), goal_handle}.detach();
//   }

//   void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
//   {
//     RCLCPP_INFO(get_logger(), "Executing goal");
//     auto result = std::make_shared<articubot_msgs::action::ArticubotTask::Result>();

//     auto move_group_node = rclcpp::Node::make_shared("move_group_interface_tutorial");
    
//     rclcpp::executors::SingleThreadedExecutor executor;
//     executor.add_node(move_group_node);
//     std::thread([&executor]() { executor.spin(); }).detach();

//     static const std::string PLANNING_GROUP = "arm_robot";
//     auto move_group_interface = std::make_shared<moveit::planning_interface::MoveGroupInterface>(move_group_node, PLANNING_GROUP);
//     auto move_group_gripper_interface = std::make_shared<moveit::planning_interface::MoveGroupInterface>(move_group_node, "gripper");

//     move_group_interface->setPlanningTime(10.0);
//     move_group_interface->setNumPlanningAttempts(10);
//     move_group_interface->setMaxVelocityScalingFactor(0.1);
//     move_group_interface->setMaxAccelerationScalingFactor(0.1);

//     std::vector<double> gripper_joint_values;
    
//     double GRIPPER_DEFAULT = to_radians(-20);
//     double GRIPPER_OPEN = to_radians(-40);
//     double GRIPPER_CLOSE = to_radians(180);

//     double goal_p_x = goal_handle->get_goal()->p_x;
//     double goal_p_y = goal_handle->get_goal()->p_y;
//     double goal_p_z = goal_handle->get_goal()->p_z;
//     double goal_or_x = goal_handle->get_goal()->or_x;
//     double goal_or_y = goal_handle->get_goal()->or_y;
//     double goal_or_z = goal_handle->get_goal()->or_z;
//     double goal_or_w = goal_handle->get_goal()->or_w;

//     geometry_msgs::msg::Pose target_pose;
//     target_pose.position.x = goal_p_x;
//     target_pose.position.y = goal_p_y;
//     target_pose.position.z = goal_p_z;
//     target_pose.orientation.x = goal_or_x;
//     target_pose.orientation.y = goal_or_y;
//     target_pose.orientation.z = goal_or_z;
//     target_pose.orientation.w = goal_or_w;

//     // Kiểm tra tính hợp lệ của mục tiêu
//     auto start_state = move_group_interface->getCurrentState();
//     RCLCPP_INFO(get_logger(), "Start state is valid: %s", start_state->satisfiesBounds() ? "true" : "false");

//     auto goal_state = std::make_shared<moveit::core::RobotState>(*start_state);
//     const moveit::core::JointModelGroup* joint_model_group = 
//       goal_state->getJointModelGroup(move_group_interface->getName());
    

//     auto current_pose = move_group_interface->getCurrentPose().pose;
//     RCLCPP_INFO(get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
//                 current_pose.position.x, current_pose.position.y, current_pose.position.z,
//                 current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
//     RCLCPP_INFO(get_logger(), "Target pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
//                 target_pose.position.x, target_pose.position.y, target_pose.position.z,
//                 target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w);



//     move_group_interface->setPoseTarget(target_pose);

//     // Lập kế hoạch và thực thi
//     moveit::planning_interface::MoveGroupInterface::Plan my_plan;
//     bool success = false;
//     std::vector<std::string> planners = { "RRTConnect","RRTstar","PRMstar","LBKPIECE","BKPIECE"};
    
//     for (const auto& planner : planners) {
//       move_group_interface->setPlannerId(planner);
//       RCLCPP_INFO(get_logger(), "Attempting to plan with %s", planner.c_str());
//       success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//       if (success) {
//         RCLCPP_INFO(get_logger(), "Planning succeeded with %s", planner.c_str());
//         break;
//       }
//     }

//     if (success) {
//       RCLCPP_INFO(get_logger(), "Lập kế hoạch thành công, thực thi...");
//         // Mở gripper
//       move_group_gripper_interface->setNamedTarget("open");
//       move_group_gripper_interface->move(); 
//       move_group_interface->execute(my_plan);
//     } else {
//       RCLCPP_ERROR(get_logger(), "Lập kế hoạch thất bại với tất cả các planner");
      
//       // Thử Cartesian path
//       RCLCPP_INFO(get_logger(), "Attempting Cartesian path planning...");
//       std::vector<geometry_msgs::msg::Pose> waypoints;
//       waypoints.push_back(move_group_interface->getCurrentPose().pose);
//       waypoints.push_back(target_pose);

//       moveit_msgs::msg::RobotTrajectory trajectory;
//       double fraction = move_group_interface->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

//       if (fraction > 0.0) {
//         RCLCPP_INFO(get_logger(), "Cartesian path (%.2f%% achieved)", fraction * 100.0);
//         my_plan.trajectory_ = trajectory;
//         success = true;


//         move_group_interface->execute(my_plan);
//       } else {
//         RCLCPP_ERROR(get_logger(), "Cartesian path planning failed");
        
//         // Ghi log chi tiết về trạng thái hiện tại và mục tiêu
//         auto current_pose = move_group_interface->getCurrentPose().pose;
//         RCLCPP_INFO(get_logger(), "Vị trí hiện tại: x=%.3f, y=%.3f, z=%.3f", 
//                     current_pose.position.x, current_pose.position.y, current_pose.position.z);
//         RCLCPP_INFO(get_logger(), "Mục tiêu: x=%.3f, y=%.3f, z=%.3f", 
//                     target_pose.position.x, target_pose.position.y, target_pose.position.z);

//         // In ra giới hạn khớp
//         for (const auto& joint_name : joint_model_group->getVariableNames()) {
//           const moveit::core::VariableBounds& bounds = goal_state->getJointModel(joint_name)->getVariableBounds()[0];
//           if (bounds.position_bounded_) {
//             RCLCPP_INFO(get_logger(), "Joint %s bounds: [%.3f, %.3f]", 
//                         joint_name.c_str(), bounds.min_position_, bounds.max_position_);
//           }
//         }

//         result->success = false;
//         goal_handle->abort(result);
//         return;
//       }
//     }

//     // Xử lý các task cụ thể (giữ nguyên phần này từ mã gốc của bạn)
//     switch (goal_handle->get_goal()->task) {
//       case 0:
      
//         // Đóng gripper
//         gripper_joint_values = {GRIPPER_CLOSE};
//         move_group_gripper_interface->setNamedTarget("closed");
//         move_group_gripper_interface->move();
        
//         // Di chuyển về home
//         move_group_interface->setNamedTarget("home");
//         move_group_interface->move();
        
//         // Mở gripper
//         gripper_joint_values = {GRIPPER_OPEN};
//         move_group_gripper_interface->setNamedTarget("open");
//         move_group_gripper_interface->move();
        
//         // Đặt gripper về vị trí mặc định
//         gripper_joint_values = {GRIPPER_DEFAULT};
//         move_group_gripper_interface->setNamedTarget("normal");
//         move_group_gripper_interface->move();
//         break;
      
//       case 1:
//         // Di chuyển về home
//         move_group_interface->setNamedTarget("home");
//         move_group_interface->move();
//         break;
      
//       case 2:
//         // Không cần thao tác gripper
//         break;
      
//       default:
//         RCLCPP_ERROR(get_logger(), "Invalid Task Number");
//         result->success = false;
//         goal_handle->abort(result);
//         return;
//     }

//     RCLCPP_INFO(get_logger(), "Planning frame: %s", move_group_interface->getPlanningFrame().c_str());
//     RCLCPP_INFO(get_logger(), "End effector link: %s", move_group_interface->getEndEffectorLink().c_str());
//     RCLCPP_INFO(get_logger(), "Available planning groups:");
//     std::vector<std::string> group_names = move_group_interface->getJointModelGroupNames();
//     for (const auto& group_name : group_names) {
//       RCLCPP_INFO(get_logger(), "  %s", group_name.c_str());
//     }

//     for (size_t i = 0; i < current_joint.size(); i++) {
//       RCLCPP_INFO(get_logger(), "Joint %ld: %f", i, current_joint[i]);
//     }

//     auto current_pose_final = move_group_interface->getCurrentPose().pose;
//     RCLCPP_INFO(get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
//                 current_pose_final.position.x, current_pose_final.position.y, current_pose_final.position.z,
//                 current_pose_final.orientation.x, current_pose_final.orientation.y, current_pose_final.orientation.z, current_pose_final.orientation.w);

//     result->success = true;
//     goal_handle->succeed(result);
//   }
// };

// }  // namespace articubot_remote

// RCLCPP_COMPONENTS_REGISTER_NODE(articubot_remote::Test_server)