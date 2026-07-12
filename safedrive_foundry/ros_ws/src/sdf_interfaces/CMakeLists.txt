cmake_minimum_required(VERSION 3.8)
project(sdf_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RunIdentity.msg"
  "msg/Frame.msg"
  "msg/EgoState.msg"
  "msg/ActorState.msg"
  "msg/Route.msg"
  "msg/TrajectoryPoint.msg"
  "msg/Trajectory.msg"
  "msg/PolicyDecision.msg"
  "msg/SafetyStatus.msg"
  "msg/ControlCommand.msg"
  "msg/RunEvent.msg"
  "srv/GetRuntimeProfile.srv"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
