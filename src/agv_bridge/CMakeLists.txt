cmake_minimum_required(VERSION 3.8)
project(agv_bridge)

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# 添加包含目录
include_directories(include)

# 创建库
add_library(${PROJECT_NAME} SHARED
  src/bridge_manager.cpp
)

# 链接依赖
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  geometry_msgs
  std_msgs
)

# 导出头文件
install(DIRECTORY include/ DESTINATION include)

# 导出库
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# 导出 cmake 配置以便其他包使用
ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})

ament_package()
