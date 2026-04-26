#!/bin/bash

# 启动AGV启动WebSocket服务器并保持运行
source /home/dts/agv_arm/install/setup.bash
python3 /home/dts/agv_arm/src/startup/src/agv_startup_websocket.py