# AGV Startup Package

此包提供了一个WebSocket服务器，用于通过WebSocket接口启动和停止AGV相关的launch文件。

## 功能

- 通过WebSocket接收启动/停止AGV系统的指令
- 支持启动`agv_bringup`包中的`agv.launch.xml`文件
- 支持远程控制AGV系统的启动和停止

## 安装

```bash
cd ~/agv_arm
colcon build --packages-select startup
source install/setup.bash
```

## 使用方法

### 启动WebSocket服务器

```bash
# 方法1: 直接运行
ros2 run startup agv_startup_websocket.py

# 方法2: 使用启动脚本
~/agv_arm/src/startup/start_agv_server.sh
```

服务器默认监听端口 `9001`，地址为 `0.0.0.0`。

### WebSocket通信协议

客户端应发送以下格式的JSON消息：

```json
{
  "target": "arm_left_start",
  "command": "true/false"
}
```

- `target`: 目标设备或功能，目前支持 `"arm_left_start"`
- `command`: 控制命令，`"true"` 启动系统，`"false"` 停止系统

### 示例

启动AGV系统：
```json
{
  "target": "arm_left_start",
  "command": "true"
}
```

停止AGV系统：
```json
{
  "target": "arm_left_start",
  "command": "false"
}
```

## 响应消息

服务器会返回以下类型的响应消息：

- `connection_ack`: 连接确认消息
- `command_ack`: 命令执行确认消息
- `error`: 错误消息

## 依赖

- Python 3
- websockets 库
- ROS2 Jazzy
- agv_bringup 包

## 故障排除

1. 如果遇到端口占用问题，请检查是否有其他实例正在运行：
   ```bash
   lsof -i :9001
   ```

2. 如果无法启动launch文件，请确保ROS2环境已正确设置：
   ```bash
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   ```