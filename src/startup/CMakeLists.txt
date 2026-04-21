cmake_minimum_required(VERSION 3.8)
project(startup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

# Install Python scripts
install(PROGRAMS
  src/agv_startup_websocket.py
  start_agv_server.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
