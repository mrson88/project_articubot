// #include <rclcpp/rclcpp.hpp>
// #include <rclcpp_action/rclcpp_action.hpp>
// #include <rclcpp_components/register_node_macro.hpp>
// #include "articubot_msgs/action/articubot_task.hpp"
// #include <moveit/move_group_interface/move_group_interface.h>
// #include "geometry_msgs/msg/quaternion.hpp"
// #include <memory>
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
// #include <string>
// #include <moveit/robot_state/robot_state.h>
// #include <moveit/robot_model/robot_model.h>
// #include <moveit/planning_scene/planning_scene.h>
//   double to_radians(const double deg_angle)
// {
//   return deg_angle * M_PI / 180.0;
// }
//  std::vector <double> current_joint = {0.0, -0.5, 1.0,1.2, 0.5,0,0};
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
//     (void)goal_handle;
//     RCLCPP_INFO(get_logger(), "Received request to cancel goal");
//     auto arm_move_group = moveit::planning_interface::MoveGroupInterface(shared_from_this(), "arm_robot");
//     auto gripper_move_group = moveit::planning_interface::MoveGroupInterface(shared_from_this(), "gripper");
//     arm_move_group.stop();
//     gripper_move_group.stop();
//     return rclcpp_action::CancelResponse::ACCEPT;
//   }
//   void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
//   {
//     // RCLCPP_INFO(this->get_logger(), "Received joint states");

//     // Access the joint positions from msg->position
//     for (size_t i = 0; i < msg->position.size(); i++)
//     {
//       // RCLCPP_INFO(this->get_logger(), "Joint %d: %f", i, msg->position[i]);
//       current_joint[i] = msg->position[i];
//     }
//   }
//   void acceptedCallback(
//       const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
//   {
//     // this needs to return quickly to avoid blocking the executor, so spin up a new thread
//     std::thread{ std::bind(&Test_server::execute, this, _1), goal_handle }.detach();
//   }

//   void execute(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)



//   {
//     RCLCPP_INFO(get_logger(), "Executing goal");
//     auto result = std::make_shared<articubot_msgs::action::ArticubotTask::Result>();

//     // MoveIt 2 Interface
//   auto const node = std::make_shared<rclcpp::Node>(
//   "test_server_moveit_robot", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

//   // Create a ROS logger
//   auto const logger = rclcpp::get_logger("test_server_moveit_robot");

//   // Create the MoveIt MoveGroup Interface
//   using moveit::planning_interface::MoveGroupInterface;
//   auto move_group_interface = MoveGroupInterface(node, "arm_robot");
//    auto move_group_gripper_interface = MoveGroupInterface(node, "gripper");

//   move_group_interface.setPlannerId("RRTConnect");
//   move_group_interface.setPlanningTime(10.0);
//   move_group_interface.setNumPlanningAttempts(10);
//   move_group_interface.setMaxVelocityScalingFactor(0.1);
//   move_group_interface.setMaxAccelerationScalingFactor(0.1);
//   std::vector<double> gripper_joint_values;
  
//   double GRIPPER_DEFAULT = to_radians(-20);
//   double GRIPPER_OPEN = to_radians(-40);
//   double GRIPPER_CLOSE = to_radians(180);

//   double goal_p_x = goal_handle->get_goal()->p_x;
//   double goal_p_y = goal_handle->get_goal()->p_y;
//   double goal_p_z = goal_handle->get_goal()->p_z;
//   double goal_or_x = goal_handle->get_goal()->or_x;
//   double goal_or_y = goal_handle->get_goal()->or_y;
//   double goal_or_z = goal_handle->get_goal()->or_z;
//   double goal_or_w = goal_handle->get_goal()->or_w;

//   // const moveit::core::JointModelGroup* joint_model_group =
//   //     move_group_interface.getCurrentState()->getJointModelGroup("arm_robot");
// switch (goal_handle->get_goal()->task){
//     case 0:
//     {
//       RCLCPP_INFO(get_logger(), "Executing goal 0");  
//       std::vector<double> gripper_joint_goal;
//       geometry_msgs::msg::Pose msg;
//       tf2::Quaternion q;
//       gripper_joint_goal = {GRIPPER_OPEN};
//       move_group_gripper_interface.setJointValueTarget(gripper_joint_goal);
//       move_group_gripper_interface.move();

//       RCLCPP_INFO(get_logger(), "Vị trí hiện tại: x=%.3f, y=%.3f, z=%.3f, orx=%.3f, ory=%.3f, orz=%.3f, orw=%.3f", 
//                   goal_p_x, goal_p_y, goal_p_z,goal_or_x,goal_or_y,goal_or_z,goal_or_w);
//       msg.position.x = goal_p_x;
//       msg.position.y = goal_p_y;
//       msg.position.z = goal_p_z;
//       msg.orientation.x = goal_or_x;
//       msg.orientation.y = goal_or_y;
//       msg.orientation.z = goal_or_z;
//       msg.orientation.w = goal_or_w;
//       move_group_interface.setPoseTarget(msg);

//       RCLCPP_INFO(get_logger(), "Finish goal 0");  
//       move_group_interface.move();



//       gripper_joint_values = {GRIPPER_CLOSE};
//       move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
//       move_group_gripper_interface.move();

//       // move_group_gripper_interface.move();
//       // move_group_interface.setNamedTarget("vertical");
//       // move_group_interface.move();
//       move_group_interface.setNamedTarget("home");
//       move_group_interface.move();
//       gripper_joint_values = {GRIPPER_OPEN};
//       move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
//       move_group_gripper_interface.move();


//       gripper_joint_values = {GRIPPER_DEFAULT};
//       move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
//       move_group_gripper_interface.move();

//       // move_group_interface.setNamedTarget("vertical");
//       // move_group_interface.move();
//       // std::vector<double> arm_joint_goal;
//       // arm_joint_goal = {0.0, -0.5, -1.0,-1.2, 0.2};

//       // move_group_interface.setJointValueTarget(arm_joint_goal);
//       // move_group_interface.move();
//       result->success = true;
//       goal_handle->succeed(result);
//       RCLCPP_INFO(get_logger(), "Goal succeeded");
//       break;
      
//     }
//     case 1:
//     {
//       std::vector<double> gripper_joint_goal;
//       geometry_msgs::msg::Pose msg;
//       tf2::Quaternion q;
//       // gripper_joint_goal = {GRIPPER_OPEN};
//       // move_group_gripper_interface.setJointValueTarget(gripper_joint_goal);
//       // move_group_gripper_interface.move();
//       // msg.position.x = goal_p_x;
//       // msg.position.y = goal_p_y;
//       // msg.position.z = goal_p_z;
//       // q.setRPY(to_radians(goal_or_x), to_radians(goal_or_y), to_radians(goal_or_z));
//       // msg.orientation = tf2::toMsg(q);
//       // move_group_interface.setPoseTarget(msg);

//       // RCLCPP_INFO(get_logger(), "Executing goal 1");  
//       // move_group_interface.move();


//       // gripper_joint_values = {GRIPPER_CLOSE};
//       // move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
//       // move_group_gripper_interface.move();

//       // move_group_gripper_interface.move();
//       // move_group_interface.setNamedTarget("vertical");
//       // move_group_interface.move();
//       move_group_interface.setNamedTarget("home");
//       move_group_interface.move();
//       // gripper_joint_values = {GRIPPER_DEFAULT};
//       // move_group_gripper_interface.setJointValueTarget(gripper_joint_values);
//       // move_group_gripper_interface.move();



//       result->success = true;
//       goal_handle->succeed(result);
//       RCLCPP_INFO(get_logger(), "Goal succeeded");
//       break;
//     }
//     case 2:
//     {
//       RCLCPP_INFO(get_logger(), "Executing goal 2"); 
//       std::vector<double> gripper_joint_goal;
//       geometry_msgs::msg::Pose msg;
//       tf2::Quaternion q;
//       RCLCPP_INFO(get_logger(), "Vị trí hiện tại: x=%.3f, y=%.3f, z=%.3f, orx=%.3f, ory=%.3f, orz=%.3f, orw=%.3f", 
//                   goal_p_x, goal_p_y, goal_p_z,goal_or_x,goal_or_y,goal_or_z,goal_or_w);
//       msg.position.x = goal_p_x;
//       msg.position.y = goal_p_y;
//       msg.position.z = goal_p_z;
//       q.setRPY(to_radians(goal_or_x), to_radians(goal_or_y), to_radians(goal_or_z));
//       msg.orientation = tf2::toMsg(q);
//       move_group_interface.setPoseTarget(msg);

//        RCLCPP_INFO(get_logger(), "Finish goal 2"); 
//       move_group_interface.move();

//       result->success = true;
//       goal_handle->succeed(result);
//       RCLCPP_INFO(get_logger(), "Goal succeeded");
//       break;
//     }
//     default:
//     {
//       RCLCPP_ERROR(get_logger(), "Invalid Task Number");
//       return;
//     }


// }


// RCLCPP_INFO(rclcpp::get_logger("move_group_interface"), "Planning frame: %s", move_group_interface.getPlanningFrame().c_str());
// RCLCPP_INFO(rclcpp::get_logger("move_group_interface"), "End effector link: %s", move_group_interface.getEndEffectorLink().c_str());
// RCLCPP_INFO(rclcpp::get_logger("move_group_interface"), "Available planning groups:");
// std::vector<std::string> group_names = move_group_interface.getJointModelGroupNames();
// for (const auto& group_name : group_names) {
//     RCLCPP_INFO(rclcpp::get_logger("move_group_interface"), "  %s", group_name.c_str());
// }

// for (size_t i = 0; i < current_joint.size(); i++)
// {
//   RCLCPP_INFO(this->get_logger(), "Joint %ld: %f", i, current_joint[i]);

// }
//   }


// };
// }  // namespace arduinobot_remote

// RCLCPP_COMPONENTS_REGISTER_NODE(articubot_remote::Test_server)













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

double to_radians(const double deg_angle)
{
  return deg_angle * M_PI / 180.0;
}

std::vector<double> current_joint = {0.0, -0.5, 1.0, 1.2, 0.5, 0, 0};

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
    subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&Test_server::joint_states_callback, this, std::placeholders::_1));
  }

private:
  rclcpp_action::Server<articubot_msgs::action::ArticubotTask>::SharedPtr action_server_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;

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

  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->position.size(); i++)
    {
      current_joint[i] = msg->position[i];
    }
  }

  void acceptedCallback(
      const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
  {
    std::thread{std::bind(&Test_server::execute, this, _1), goal_handle}.detach();
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

    move_group_interface->setPlanningTime(10.0);
    move_group_interface->setNumPlanningAttempts(10);
    move_group_interface->setMaxVelocityScalingFactor(0.1);
    move_group_interface->setMaxAccelerationScalingFactor(0.1);

    std::vector<double> gripper_joint_values;
    
    double GRIPPER_DEFAULT = to_radians(-20);
    double GRIPPER_OPEN = to_radians(-40);
    double GRIPPER_CLOSE = to_radians(180);

    double goal_p_x = goal_handle->get_goal()->p_x;
    double goal_p_y = goal_handle->get_goal()->p_y;
    double goal_p_z = goal_handle->get_goal()->p_z;
    double goal_or_x = goal_handle->get_goal()->or_x;
    double goal_or_y = goal_handle->get_goal()->or_y;
    double goal_or_z = goal_handle->get_goal()->or_z;
    double goal_or_w = goal_handle->get_goal()->or_w;

    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = goal_p_x;
    target_pose.position.y = goal_p_y;
    target_pose.position.z = goal_p_z;
    target_pose.orientation.x = goal_or_x;
    target_pose.orientation.y = goal_or_y;
    target_pose.orientation.z = goal_or_z;
    target_pose.orientation.w = goal_or_w;

    // Kiểm tra tính hợp lệ của mục tiêu
    auto start_state = move_group_interface->getCurrentState();
    RCLCPP_INFO(get_logger(), "Start state is valid: %s", start_state->satisfiesBounds() ? "true" : "false");

    auto goal_state = std::make_shared<moveit::core::RobotState>(*start_state);
    const moveit::core::JointModelGroup* joint_model_group = 
      goal_state->getJointModelGroup(move_group_interface->getName());
    

    auto current_pose = move_group_interface->getCurrentPose().pose;
    RCLCPP_INFO(get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
                current_pose.position.x, current_pose.position.y, current_pose.position.z,
                current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
    RCLCPP_INFO(get_logger(), "Target pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
                target_pose.position.x, target_pose.position.y, target_pose.position.z,
                target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w);

    // for (const auto& joint_name : joint_model_group->getVariableNames()) {
    //     const moveit::core::VariableBounds& bounds = current_state->getJointModel(joint_name)->getVariableBounds()[0];
    //     if (bounds.position_bounded_) {
    //         RCLCPP_INFO(get_logger(), "Joint %s bounds: [%.3f, %.3f]", 
    //                     joint_name.c_str(), bounds.min_position_, bounds.max_position_);
    //     }
    // }       

//     geometry_msgs::msg::Pose near_target = current_pose;
//     near_target.position.x += 0.01;  // Di chuyển 10cm theo trục x   

//       RCLCPP_INFO(get_logger(), "Target pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
//                 near_target.position.x, near_target.position.y, near_target.position.z,
//                 near_target.orientation.x, near_target.orientation.y, near_target.orientation.z, near_target.orientation.w);
//     // bool found_ik = goal_state->setFromIK(joint_model_group, near_target);     
//     // if (!found_ik) {
//     //   RCLCPP_ERROR(get_logger(), "Không tìm thấy giải pháp IK cho mục tiêu");
//     //   result->success = false;
//     //   goal_handle->abort(result);
//     //   return;
//     // }

//     // RCLCPP_INFO(get_logger(), "Goal state is valid: %s", goal_state->satisfiesBounds() ? "true" : "false");

// bool success_appro = move_group_interface->setApproximateJointValueTarget(target_pose);
// if (success_appro) {
//     RCLCPP_INFO(get_logger(), "Successfully set approximate joint value target");
// } else {
//     RCLCPP_ERROR(get_logger(), "Failed to set approximate joint value target");
// }
// collision_detection::CollisionRequest collision_request;
// collision_detection::CollisionResult collision_result;
// planning_scene::PlanningScenePtr planning_scene = std::make_shared<planning_scene::PlanningScene>(move_group_interface->getRobotModel());
// planning_scene->checkCollision(collision_request, collision_result, *goal_state);
// if (collision_result.collision) {
//     RCLCPP_WARN(get_logger(), "Current state is in collision");
//         for (const auto& contact : collision_result.contacts) {
//         RCLCPP_INFO(get_logger(), "Collision between: %s and %s", contact.first.first.c_str(), contact.first.second.c_str());
//     }
// }
// std::vector<double> current_joint_values;
// goal_state->copyJointGroupPositions(joint_model_group, current_joint_values);
// for (size_t i = 0; i < current_joint_values.size(); ++i) {
//     RCLCPP_INFO(get_logger(), "Joint %zu: %.3f", i, current_joint_values[i]);
// }


    move_group_interface->setPoseTarget(target_pose);

    // Lập kế hoạch và thực thi
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = false;
    std::vector<std::string> planners = { "RRTConnect"};
    
    for (const auto& planner : planners) {
      move_group_interface->setPlannerId(planner);
      RCLCPP_INFO(get_logger(), "Attempting to plan with %s", planner.c_str());
      success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      if (success) {
        RCLCPP_INFO(get_logger(), "Planning succeeded with %s", planner.c_str());
        break;
      }
    }

    if (success) {
      RCLCPP_INFO(get_logger(), "Lập kế hoạch thành công, thực thi...");
      move_group_interface->execute(my_plan);
    } else {
      RCLCPP_ERROR(get_logger(), "Lập kế hoạch thất bại với tất cả các planner");
      
      // Thử Cartesian path
      RCLCPP_INFO(get_logger(), "Attempting Cartesian path planning...");
      std::vector<geometry_msgs::msg::Pose> waypoints;
      waypoints.push_back(move_group_interface->getCurrentPose().pose);
      waypoints.push_back(target_pose);

      moveit_msgs::msg::RobotTrajectory trajectory;
      double fraction = move_group_interface->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

      if (fraction > 0.0) {
        RCLCPP_INFO(get_logger(), "Cartesian path (%.2f%% achieved)", fraction * 100.0);
        my_plan.trajectory_ = trajectory;
        success = true;
        move_group_interface->execute(my_plan);
      } else {
        RCLCPP_ERROR(get_logger(), "Cartesian path planning failed");
        
        // Ghi log chi tiết về trạng thái hiện tại và mục tiêu
        auto current_pose = move_group_interface->getCurrentPose().pose;
        RCLCPP_INFO(get_logger(), "Vị trí hiện tại: x=%.3f, y=%.3f, z=%.3f", 
                    current_pose.position.x, current_pose.position.y, current_pose.position.z);
        RCLCPP_INFO(get_logger(), "Mục tiêu: x=%.3f, y=%.3f, z=%.3f", 
                    target_pose.position.x, target_pose.position.y, target_pose.position.z);

        // In ra giới hạn khớp
        for (const auto& joint_name : joint_model_group->getVariableNames()) {
          const moveit::core::VariableBounds& bounds = goal_state->getJointModel(joint_name)->getVariableBounds()[0];
          if (bounds.position_bounded_) {
            RCLCPP_INFO(get_logger(), "Joint %s bounds: [%.3f, %.3f]", 
                        joint_name.c_str(), bounds.min_position_, bounds.max_position_);
          }
        }

        result->success = false;
        goal_handle->abort(result);
        return;
      }
    }

    // Xử lý các task cụ thể (giữ nguyên phần này từ mã gốc của bạn)
    switch (goal_handle->get_goal()->task) {
      case 0:
        // Mở gripper
        gripper_joint_values = {GRIPPER_OPEN};
        move_group_gripper_interface->setJointValueTarget(gripper_joint_values);
        move_group_gripper_interface->move();
        
        // Đóng gripper
        gripper_joint_values = {GRIPPER_CLOSE};
        move_group_gripper_interface->setJointValueTarget(gripper_joint_values);
        move_group_gripper_interface->move();
        
        // Di chuyển về home
        move_group_interface->setNamedTarget("home");
        move_group_interface->move();
        
        // Mở gripper
        gripper_joint_values = {GRIPPER_OPEN};
        move_group_gripper_interface->setJointValueTarget(gripper_joint_values);
        move_group_gripper_interface->move();
        
        // Đặt gripper về vị trí mặc định
        gripper_joint_values = {GRIPPER_DEFAULT};
        move_group_gripper_interface->setJointValueTarget(gripper_joint_values);
        move_group_gripper_interface->move();
        break;
      
      case 1:
        // Di chuyển về home
        move_group_interface->setNamedTarget("home");
        move_group_interface->move();
        break;
      
      case 2:
        // Không cần thao tác gripper
        break;
      
      default:
        RCLCPP_ERROR(get_logger(), "Invalid Task Number");
        result->success = false;
        goal_handle->abort(result);
        return;
    }

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

    auto current_pose_final = move_group_interface->getCurrentPose().pose;
    RCLCPP_INFO(get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
                current_pose_final.position.x, current_pose_final.position.y, current_pose_final.position.z,
                current_pose_final.orientation.x, current_pose_final.orientation.y, current_pose_final.orientation.z, current_pose_final.orientation.w);

    result->success = true;
    goal_handle->succeed(result);
  }
};

}  // namespace articubot_remote

RCLCPP_COMPONENTS_REGISTER_NODE(articubot_remote::Test_server)