cmake_minimum_required(VERSION 3.8)
project(camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)

# Orbbec SDK path
set(ORBBEC_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/sdk)

# OpenNI2 include paths
include_directories(${ORBBEC_SDK_PATH}/Include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library search path
link_directories(${ORBBEC_SDK_PATH}/libs)

add_executable(camera_node
  src/nodes/camera_node.cpp
  src/astras/astras_camera.cpp
  src/nodes/astras_node.cpp
)
ament_target_dependencies(camera_node 
    rclcpp 
    sensor_msgs 
    cv_bridge 
    image_transport
    tf2
    tf2_ros
    geometry_msgs
)
# 4. 链接 OpenCV 和 Orbbec 的 OpenNI2 库
# 使用绝对路径链接库，可以避免一些链接搜索问题
# Link OpenCV and Orbbec OpenNI2 libraries
target_link_libraries(camera_node 
    ${OpenCV_LIBS}
    ${ORBBEC_SDK_PATH}/libs/libOpenNI2.so
)
# 5. 设置 RPATH，让可执行文件在运行时能自动找到 SDK 中的 .so 文件
# 这样你就不需要每次都手动 export LD_LIBRARY_PATH 了
# Set RPATH so the executable can find the library at runtime
set_target_properties(camera_node PROPERTIES
    INSTALL_RPATH "${ORBBEC_SDK_PATH}/libs"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS camera_node
    DESTINATION lib/${PROJECT_NAME})

# 安装配置文件
install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

# 安装 launch 文件
install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
