#!/bin/bash

# 启动AGV启动WebSocket服务器
echo "启动AGV Startup WebSocket服务器..."

# 检查ROS2环境是否已设置
if [ -z "$ROS_DISTRO" ]; then
    echo "警告: ROS2环境可能未设置，请确保已source ROS2环境"
fi

# 运行WebSocket服务器
python3 $(dirname "$0")/../src/agv_startup_websocket.py