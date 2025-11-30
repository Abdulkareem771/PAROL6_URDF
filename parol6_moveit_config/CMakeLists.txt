cmake_minimum_required(VERSION 3.5)
project(parol6_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch rviz scripts
  DESTINATION share/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

ament_package()
