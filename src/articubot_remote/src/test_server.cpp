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
// #include <std_msgs/msg/string.hpp>  
//   double to_radians(const double deg_angle)
// {
//   return deg_angle * M_PI / 180.0;
// }

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
//         this, "test_server", 
//         std::bind(&Test_server::goalCallback, this, std::placeholders::_1, std::placeholders::_2),
//         std::bind(&Test_server::cancelCallback, this, std::placeholders::_1),
//         std::bind(&Test_server::acceptedCallback, this, std::placeholders::_1));
//     subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
//         "/joint_states", 10, std::bind(&Test_server::joint_states_callback, this, std::placeholders::_1));


//     // Create the publisher for the find_ball topic
//     find_ball_publisher_ = this->create_publisher<std_msgs::msg::String>("find_ball", 10);
    

//   }

// private:
//   rclcpp_action::Server<articubot_msgs::action::ArticubotTask>::SharedPtr action_server_;
//   rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
//   rclcpp::Publisher<std_msgs::msg::String>::SharedPtr find_ball_publisher_;  
//   std::vector<double> current_joint = {0.0, -0.5, 1.0, 1.2, 0.5, 0, 0};

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
//   void send_find_ball_message(bool success)
//   {
//     auto message = std_msgs::msg::String();
//     message.data = success ? "true" : "false";
//     RCLCPP_INFO(this->get_logger(), "Publishing find_ball message: '%s'", message.data.c_str());
//     find_ball_publisher_->publish(message);
//   }
//   void acceptedCallback(
//       const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>> goal_handle)
//   {
//     std::thread{std::bind(&Test_server::execute, this, std::placeholders::_1), goal_handle}.detach();
//   }

//   void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
//   {
//     for (size_t i = 0; i < msg->position.size(); i++)
//     {
//       current_joint[i] = msg->position[i];
//     }
//   }
//   void handle_accepted(const std::shared_ptr<GoalHandleArticubotTask> goal_handle)
//   {
//     std::unique_lock<std::mutex> lock(this->mutex_);
//     this->goal_queue_.push(goal_handle);
//     this->cv_.notify_one();
//   }

//   void executorThread()
//   {
//     while (rclcpp::ok()) {
//       std::unique_lock<std::mutex> lock(this->mutex_);
//       this->cv_.wait(lock, [this] { return !this->goal_queue_.empty() || this->stop_executor_; });

//       if (this->stop_executor_) {
//         break;
//       }

//       auto goal_handle = this->goal_queue_.front();
//       this->goal_queue_.pop();
//       lock.unlock();

//       execute(goal_handle);
//     }
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

//     setupMoveGroupInterface(move_group_interface);

//     bool success = false;
//     switch (goal_handle->get_goal()->task) {
//       case 0:
//         success = openGripper(move_group_gripper_interface);
//         success = moveToPoint(move_group_interface, move_group_gripper_interface, goal_handle);
//         success = closeGripper(move_group_gripper_interface);
//         success = moveToHome(move_group_interface);
//         success = openGripper(move_group_gripper_interface);
//         break;
//       case 1:
//         success = moveToHome(move_group_interface);
//         break;
//       case 2:
//         success = closeGripper(move_group_gripper_interface);
//         break;
//       case 3:
//         success = openGripper(move_group_gripper_interface);
//         break;
//       case 4:
//         success = moveToPoint(move_group_interface, move_group_gripper_interface, goal_handle);
//         break;
//       case 5:
//         success = pickUp(move_group_interface, move_group_gripper_interface, goal_handle);
//         break;
//       default:
//         RCLCPP_ERROR(get_logger(), "Invalid Task Number");
//         result->success = false;
//         goal_handle->abort(result);
//         return;
//     }

//     if (success) {
//       logRobotState(move_group_interface);
//       result->success = true;
//       goal_handle->succeed(result);
//     } else {
//       RCLCPP_ERROR(get_logger(), "Task execution failed");
//       result->success = false;
//       goal_handle->abort(result);
//       send_find_ball_message(false);
//     }
//     send_find_ball_message(success);
//   }

//   void setupMoveGroupInterface(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
//   {
//     move_group_interface->setPlanningTime(10.0);
//     move_group_interface->setNumPlanningAttempts(10);
//     move_group_interface->setMaxVelocityScalingFactor(0.1);
//     move_group_interface->setMaxAccelerationScalingFactor(0.1);
//   }

//   bool moveToPoint(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
//                    const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
//                    const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>>& goal_handle)
//   {
//     geometry_msgs::msg::Pose target_pose = createTargetPose(goal_handle);
//     return planAndExecute(move_group_interface, move_group_gripper_interface, target_pose);
//   }
//   bool pickUp(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
//                    const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
//                    const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>>& goal_handle)
//   {
//     geometry_msgs::msg::Pose target_pose = createTargetPosePickup(goal_handle);
//     return planAndExecute(move_group_interface, move_group_gripper_interface, target_pose);
//   }
//   bool moveToHome(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
//   {
//     RCLCPP_INFO(get_logger(), "Moving to home position...");
//     move_group_interface->setNamedTarget("home");
//     return (move_group_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
//   }

//   bool closeGripper(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface)
//   {
//     RCLCPP_INFO(get_logger(), "Closing gripper...");
//     move_group_gripper_interface->setNamedTarget("closed");
//     return (move_group_gripper_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
//   }

//   bool openGripper(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface)
//   {
//     RCLCPP_INFO(get_logger(), "Closing gripper...");
//     move_group_gripper_interface->setNamedTarget("open");
//     return (move_group_gripper_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
//   }

//   geometry_msgs::msg::Pose createTargetPose(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>>& goal_handle)
//   {
//     geometry_msgs::msg::Pose target_pose;
//     target_pose.position.x = goal_handle->get_goal()->p_x;
//     target_pose.position.y = goal_handle->get_goal()->p_y;
//     target_pose.position.z = goal_handle->get_goal()->p_z;
//     target_pose.orientation.x = goal_handle->get_goal()->or_x;
//     target_pose.orientation.y = goal_handle->get_goal()->or_y;
//     target_pose.orientation.z = goal_handle->get_goal()->or_z;
//     target_pose.orientation.w = goal_handle->get_goal()->or_w;
//     return target_pose;
//   }
//   geometry_msgs::msg::Pose createTargetPosePickup(const std::shared_ptr<rclcpp_action::ServerGoalHandle<articubot_msgs::action::ArticubotTask>>& goal_handle)
//   {
//     geometry_msgs::msg::Pose target_pose;
//     tf2::Quaternion q;

//     target_pose.position.x = goal_handle->get_goal()->p_x;
//     target_pose.position.y = goal_handle->get_goal()->p_y;
//     target_pose.position.z = goal_handle->get_goal()->p_z;
//     double goal_or_x = goal_handle->get_goal()->or_x;
//     double goal_or_y = goal_handle->get_goal()->or_y;
//     double goal_or_z = goal_handle->get_goal()->or_z;
//     q.setRPY(to_radians(goal_or_x), to_radians(goal_or_y), to_radians(goal_or_z));
//     target_pose.orientation = tf2::toMsg(q);
//     return target_pose;
//   }
//   bool planAndExecute(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
//                       const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
//                       const geometry_msgs::msg::Pose& target_pose)
//   {
//     move_group_interface->setPoseTarget(target_pose);
//     moveit::planning_interface::MoveGroupInterface::Plan my_plan;
//     bool success = false;

//     // std::vector<std::string> planners = {"RRTConnect", "RRTstar", "PRMstar", "LBKPIECE", "BKPIECE"};
//     std::vector<std::string> planners = {"RRTConnect",};   
//     for (const auto& planner : planners) {
//       move_group_interface->setPlannerId(planner);
//       RCLCPP_INFO(get_logger(), "Attempting to plan with %s", planner.c_str());
//       success = (move_group_interface->plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//       if (success) {
//         RCLCPP_INFO(get_logger(), "Planning succeeded with %s", planner.c_str());
//         break;
//       }
//     }

//     if (!success) {
//       RCLCPP_ERROR(get_logger(), "Planning failed with all planners");
//       success = attemptCartesianPath(move_group_interface, target_pose);
//     }

//     if (success) {
//       RCLCPP_INFO(get_logger(), "Execution starting...");
//       move_group_gripper_interface->setNamedTarget("open");
//       move_group_gripper_interface->move();
//       success = (move_group_interface->execute(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
//       if (!success) {
//         RCLCPP_ERROR(get_logger(), "Execution failed");
//       }
//     }

//     return success;
//   }

//   bool attemptCartesianPath(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
//                             const geometry_msgs::msg::Pose& target_pose)
//   {
//     RCLCPP_INFO(get_logger(), "Attempting Cartesian path planning...");
//     std::vector<geometry_msgs::msg::Pose> waypoints;
//     waypoints.push_back(move_group_interface->getCurrentPose().pose);
//     waypoints.push_back(target_pose);

//     moveit_msgs::msg::RobotTrajectory trajectory;
//     double fraction = move_group_interface->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

//     if (fraction > 0.0) {
//       RCLCPP_INFO(get_logger(), "Cartesian path (%.2f%% achieved)", fraction * 100.0);
//       return (move_group_interface->execute(trajectory) == moveit::core::MoveItErrorCode::SUCCESS);
//     } else {
//       RCLCPP_ERROR(get_logger(), "Cartesian path planning failed");
//       logPlanningFailure(move_group_interface, target_pose);
//       return false;
//     }
//   }

//   void logPlanningFailure(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
//                           const geometry_msgs::msg::Pose& target_pose)
//   {
//     auto current_pose = move_group_interface->getCurrentPose().pose;
//     RCLCPP_INFO(get_logger(), "Current position: x=%.3f, y=%.3f, z=%.3f", 
//                 current_pose.position.x, current_pose.position.y, current_pose.position.z);
//     RCLCPP_INFO(get_logger(), "Target position: x=%.3f, y=%.3f, z=%.3f", 
//                 target_pose.position.x, target_pose.position.y, target_pose.position.z);

//     const moveit::core::JointModelGroup* joint_model_group = 
//       move_group_interface->getCurrentState()->getJointModelGroup(move_group_interface->getName());
    
//     for (const auto& joint_name : joint_model_group->getVariableNames()) {
//       const moveit::core::VariableBounds& bounds = move_group_interface->getCurrentState()->getJointModel(joint_name)->getVariableBounds()[0];
//       if (bounds.position_bounded_) {
//         RCLCPP_INFO(get_logger(), "Joint %s bounds: [%.3f, %.3f]", 
//                     joint_name.c_str(), bounds.min_position_, bounds.max_position_);
//       }
//     }
//   }

//   void logRobotState(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
//   {
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

//     auto current_pose = move_group_interface->getCurrentPose().pose;
//     RCLCPP_INFO(get_logger(), "Current pose: x=%.3f, y=%.3f, z=%.3f, ox=%.3f, oy=%.3f, oz=%.3f, ow=%.3f",
//                 current_pose.position.x, current_pose.position.y, current_pose.position.z,
//                 current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w);
//   }
// };

// }  // namespace articubot_remote

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
#include <std_msgs/msg/string.hpp>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <chrono>
#include <atomic>

namespace articubot_remote
{

class Test_server : public rclcpp::Node
{
public:
  using ArticubotTask = articubot_msgs::action::ArticubotTask;
  using GoalHandleArticubotTask = rclcpp_action::ServerGoalHandle<ArticubotTask>;

  explicit Test_server(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("test_server", options)
  {
    using namespace std::placeholders;

    this->action_server_ = rclcpp_action::create_server<ArticubotTask>(
      this,
      "test_server",
      std::bind(&Test_server::handle_goal, this, _1, _2),
      std::bind(&Test_server::handle_cancel, this, _1),
      std::bind(&Test_server::handle_accepted, this, _1));

    this->subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, std::bind(&Test_server::joint_states_callback, this, _1));

    this->find_ball_publisher_ = this->create_publisher<std_msgs::msg::String>("find_ball", 10);

    this->executor_thread_ = std::thread(&Test_server::executorThread, this);

    this->executor_check_timer_ = this->create_wall_timer(
      std::chrono::seconds(5),
      std::bind(&Test_server::checkExecutorStatus, this));
  }

  ~Test_server()
  {
    {
      std::unique_lock<std::mutex> lock(this->mutex_);
      this->stop_executor_ = true;
    }
    this->cv_.notify_one();
    if (this->executor_thread_.joinable()) {
      this->executor_thread_.join();
    }
  }

private:
  rclcpp_action::Server<ArticubotTask>::SharedPtr action_server_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr find_ball_publisher_;
  rclcpp::TimerBase::SharedPtr executor_check_timer_;
  std::vector<double> current_joint = {0.0, -0.5, 1.0, 1.2, 0.5, 0, 0};

  std::thread executor_thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::shared_ptr<GoalHandleArticubotTask> current_goal_;
  bool stop_executor_ = false;
  std::atomic<bool> is_executing_{false};
  std::atomic<bool> stop_execution_{false};
  const std::chrono::seconds TIMEOUT_DURATION{20}; // Timeout sau 60 giây

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const ArticubotTask::Goal> goal)
  {
    RCLCPP_INFO(this->get_logger(), "Received goal request with order %d", goal->task);
    (void)uuid;
    if (is_executing_) {
      RCLCPP_WARN(this->get_logger(), "Rejecting new goal, currently executing another goal");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandleArticubotTask> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Received request to cancel goal");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleArticubotTask> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Goal accepted, starting execution");
    std::unique_lock<std::mutex> lock(this->mutex_);
    current_goal_ = goal_handle;
    is_executing_ = true;
    this->cv_.notify_one();
  }

  void executorThread()
  {
    RCLCPP_INFO(this->get_logger(), "Executor thread started");
    while (rclcpp::ok()) {
      std::unique_lock<std::mutex> lock(this->mutex_);
      this->cv_.wait(lock, [this] { return current_goal_ != nullptr || this->stop_executor_; });

      if (this->stop_executor_) {
        RCLCPP_INFO(this->get_logger(), "Executor thread stopping");
        break;
      }

      if (current_goal_) {
        auto goal_handle = current_goal_;
        lock.unlock();

        RCLCPP_INFO(this->get_logger(), "Starting to process goal");
        execute(goal_handle);

        lock.lock();
        current_goal_ = nullptr;
        is_executing_ = false;
        RCLCPP_INFO(this->get_logger(), "Finished processing goal");
      }
    }
    RCLCPP_INFO(this->get_logger(), "Executor thread ended");
  }

  void execute(const std::shared_ptr<GoalHandleArticubotTask> goal_handle)
  {
    RCLCPP_INFO(get_logger(), "Starting goal execution");
    auto result = std::make_shared<ArticubotTask::Result>();

    stop_execution_ = false;
    auto future = std::async(std::launch::async, [this, goal_handle]() {
      return executeGoal(goal_handle);
    });

    if (future.wait_for(TIMEOUT_DURATION) == std::future_status::timeout) {
      RCLCPP_ERROR(this->get_logger(), "Goal execution timed out");
      stop_execution_ = true;
      std::this_thread::sleep_for(std::chrono::seconds(1));
      result->success = false;
      goal_handle->abort(result);
    } else {
      result->success = future.get();
      if (result->success) {
        goal_handle->succeed(result);
      } else {
        goal_handle->abort(result);
      }
    }

    send_find_ball_message(result->success);
    RCLCPP_INFO(get_logger(), "Goal execution completed with result: %s", result->success ? "success" : "failure");
  }

  bool executeGoal(const std::shared_ptr<GoalHandleArticubotTask> goal_handle)
  {
    auto move_group_node = std::make_shared<rclcpp::Node>("move_group_interface_tutorial");
    
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
          bool success1 = openGripper(move_group_gripper_interface);
          bool success2 = moveToPoint(move_group_interface, move_group_gripper_interface, goal_handle);
          bool success3 = closeGripper(move_group_gripper_interface);
          bool success4 = moveToHome(move_group_interface);
          bool success5 = openGripper(move_group_gripper_interface);
          success=success1 && success2 && success2=3 && success4 && success5
        break;
      case 1:
        success = moveToHome(move_group_interface);
        break;
      case 2:
        success = closeGripper(move_group_gripper_interface);
        break;
      case 3:
        success = openGripper(move_group_gripper_interface);
        break;
      case 4:
        success = moveToPoint(move_group_interface, move_group_gripper_interface, goal_handle);
        break;
      case 5:
        success = pickUp(move_group_interface, move_group_gripper_interface, goal_handle);
        break;
      default:
        RCLCPP_ERROR(get_logger(), "Invalid Task Number");
        success = false;
    }

    if (success) {
      logRobotState(move_group_interface);
    }

    return success;
  }

  template<typename Func, typename... Args>
  bool executeWithTimeout(Func&& func, Args&&... args)
  {
    if (stop_execution_) return false;
    auto result = func(std::forward<Args>(args)...);
    return result && !stop_execution_;
  }

  template<typename... Funcs>
  bool executeTaskSequence(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                           const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
                           Funcs... funcs)
  {
    return (... && executeWithTimeout(funcs));
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
                   const std::shared_ptr<GoalHandleArticubotTask>& goal_handle)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Moving to point");
    geometry_msgs::msg::Pose target_pose = createTargetPose(goal_handle);
    return planAndExecute(move_group_interface, move_group_gripper_interface, target_pose);
  }

  bool pickUp(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
              const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
              const std::shared_ptr<GoalHandleArticubotTask>& goal_handle)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Starting pickUp operation");
    geometry_msgs::msg::Pose target_pose = createTargetPosePickup(goal_handle);
    RCLCPP_INFO(get_logger(), "Target pose: x=%.3f, y=%.3f, z=%.3f", 
                target_pose.position.x, target_pose.position.y, target_pose.position.z);
    
    bool success = planAndExecute(move_group_interface, move_group_gripper_interface, target_pose);
    
    if (success) {
        RCLCPP_INFO(get_logger(), "pickUp operation completed successfully");
    } else {
        RCLCPP_ERROR(get_logger(), "pickUp operation failed");
    }
    
    return success;
  }

  bool moveToHome(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Moving to home position...");
    move_group_interface->setNamedTarget("home");
    return (move_group_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool closeGripper(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Closing gripper...");
    move_group_gripper_interface->setNamedTarget("closed");
    return (move_group_gripper_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

  bool openGripper(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Opening gripper...");
    move_group_gripper_interface->setNamedTarget("open");
    return (move_group_gripper_interface->move() == moveit::core::MoveItErrorCode::SUCCESS);
  }

geometry_msgs::msg::Pose createTargetPose(const std::shared_ptr<GoalHandleArticubotTask>& goal_handle)
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

  geometry_msgs::msg::Pose createTargetPosePickup(const std::shared_ptr<GoalHandleArticubotTask>& goal_handle)
  {
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = goal_handle->get_goal()->p_x;
    target_pose.position.y = goal_handle->get_goal()->p_y;
    target_pose.position.z = goal_handle->get_goal()->p_z;
    target_pose.orientation.w = 1;
    return target_pose;
  }

  bool planAndExecute(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                      const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_gripper_interface,
                      const geometry_msgs::msg::Pose& target_pose)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Planning and executing movement");
    move_group_interface->setPoseTarget(target_pose);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool success = false;

    std::vector<std::string> planners = {"RRTConnect"};   
    for (const auto& planner : planners) {
      if (stop_execution_) return false;
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

    if (success && !stop_execution_) {
      RCLCPP_INFO(get_logger(), "Execution starting...");
      move_group_gripper_interface->setNamedTarget("open");
      move_group_gripper_interface->move();
      success = (move_group_interface->execute(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
      if (!success) {
        RCLCPP_ERROR(get_logger(), "Execution failed");
      }
    }

    return success && !stop_execution_;
  }

  bool attemptCartesianPath(const std::shared_ptr<moveit::planning_interface::MoveGroupInterface>& move_group_interface,
                            const geometry_msgs::msg::Pose& target_pose)
  {
    if (stop_execution_) return false;
    RCLCPP_INFO(get_logger(), "Attempting Cartesian path planning...");
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(move_group_interface->getCurrentPose().pose);
    waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    double fraction = move_group_interface->computeCartesianPath(waypoints, 0.01, 0.0, trajectory);

    if (fraction > 0.0 && !stop_execution_) {
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

  void send_find_ball_message(bool success)
  {
    auto message = std_msgs::msg::String();
    message.data = success ? "true" : "false";
    RCLCPP_INFO(this->get_logger(), "Publishing find_ball message: '%s'", message.data.c_str());
    find_ball_publisher_->publish(message);
  }

  void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    for (size_t i = 0; i < msg->position.size(); i++)
    {
      current_joint[i] = msg->position[i];
    }
  }

  void checkExecutorStatus()
  {
    std::unique_lock<std::mutex> lock(this->mutex_);
    RCLCPP_INFO(this->get_logger(), "Executor status: Is executing goal: %s, Current goal: %s",
                is_executing_ ? "Yes" : "No", current_goal_ ? "Present" : "None");
  }
};

}  // namespace articubot_remote

RCLCPP_COMPONENTS_REGISTER_NODE(articubot_remote::Test_server)