# 下位机端通信及控制说明

## 各文件功能说明：

### agv\_hardware.hpp:实现ROS2 hardwareinterface::SystemInterface 接口\_

### config.hpp:包含所有可配置参数，包括机械臂网络配置，WebSocket配置，初始化状态配置

### robot\_controller.hpp:封装与珞石机械臂相关的通信和控制

### websocket\_clint.hpp:实现WebSocket客户端，发送关节位置数据到上位机

### websocket\_server.hpp:实现WebSocket服务端，接收上位机控制指令

### command\_parser.hpp:解析与构建JSON格式的命令和响应

```
┌─────────────────────────┐
│  AgvHardwareInterface   │  // 硬件接口核心
├─────────────────────────┤
│  ┌─────────────────────┐│
│  │  RobotController    ││  // 机器人控制
│  ├─────────────────────┤│
│  │  WebSocketClient    ││  // 与上位机通信
│  ├─────────────────────┤│
│  │  WebSocketServer    ││  // 本地控制
│  └─────────────────────┘│
│  ┌─────────────────────┐│
│  │  CommandParser      ││  // 命令解析
│  └─────────────────────┘│
└─────────────────────────┘
```

## 设备接入说明 config.hpp

在结构体AgvConfig中引入要连接的设备，如下：

```cpp
robot_ip = "192.168.1.159";
```

## agv\_*hardwareinterface.cpp函数功能详解*

## agv设备初始化配置&#x20;

在on\_init()函数中引入要初始化连接的设备，如下：

```C++
// 初始化配置
    config_ = AgvConfig();
    
    // 初始化机器人控制器
    robot_controller_ = std::make_unique<RobotController>(config_);
    
    // 初始化 WebSocket 客户端
    ws_client_ = std::make_unique<WebSocketClientWrapper>(
        config_.server_uri, 
        config_.reconnect_interval_ms
    );
```

主要包括，设备ip初始化以及websocket通信机制建立

## 设备状态参数配置

在on\_configure()和on\_activate()函数中连接设备以及配置初始状态，上位机连接下默认进入非实时状态，具体状态配置分在robot\_controller.cpp文件中

```C++
if (!robot_controller_->activate()) {
        return hardware_interface::CallbackReturn::ERROR;
    }
```

网络连接激活，启动本地的服务端和客户端，一方主动连接，另一方被动接收

```C++
// 启动 WebSocket 客户端
    ws_client_->start();
    
    // 启动本地 WebSocket 服务器
    ws_server_->start();
```

## 设备失能

在on\_deactive()函数中，关闭WebSocket网络通信，机械臂下电，具体的实现函数在websocketclient.cpp和websocketserver.cpp和robot\_\_controller.cpp中

```C++
// 停止 WebSocket 通信
    ws_server_->stop();
    ws_client_->stop();
    
    // 停用机器人控制器
    robot_controller_->deactivate();
```

## 信息回传

在read()函数中，将机械臂的位姿，关节角，agv的位置信息上传至上位机

```C++
// 发送关节位置到 WebSocket 服务器（新协议格式）
    reportJointPositions(joint_positions_);
```

## 实时模式下机械臂控制接口

在write()函数中，通过写入关节角改变机械臂姿态

```C++
// 如果是实时模式，发送命令到机器人控制器
    if (config_.rt_mode) {
        robot_controller_->sendJointCommand(joint_commands_);
    }
```

## 关节信息封装函数

在reportJointPositions()函数中，具体实现在command\_parser.cpp文件中

```cpp
// 使用新协议格式构建状态消息
        std::string json_str = command_parser_.buildStatusMessage(positions, "mobile_dual_arm_robot");
        
        // 发送数据
        ws_client_->send(json_str);
```

## 指令执行

<br />

在handleCommand()函数中，解析控制指令，并调用对应的SDK函数控制机械臂

```C++
if (arm_cmd.arm_id == "left_arm") {
                if (arm_cmd.is_joint_command) {
                    // 关节空间命令
                    if (config_.rt_mode) {
                        // 实时模式：设置命令，由 write() 中的 RT 循环执行
                        std::lock_guard<std::mutex> lock(io_mutex_);
                        joint_commands_ = arm_cmd.parameters;
                        std::cout << "[AgvHardware] 实时模式：关节命令已缓存" << std::endl;
                    } else {
                        // 非实时模式：直接发送运动命令
                        robot_controller_->sendJointCommand(arm_cmd.parameters);
                        
                        if (cmd->header.msg_type == "command") {
                            robot_controller_->waitForMotionComplete();
                        }
                    }
                } else {
                    // 笛卡尔空间命令
                    if (config_.rt_mode) {
                        std::cerr << "[AgvHardware] 实时模式暂不支持笛卡尔空间控制" << std::endl;
                        all_success = false;
                    } else {
                        // 非实时模式：直接发送运动命令
                        std::vector<std::array<double, 6>> points;
                        points.push_back(arm_cmd.parameters);
                        std::string move_type = (arm_cmd.command == "cartesian_move") ? "linear" : "joint";
                        robot_controller_->sendCartesianCommand(points, move_type, arm_cmd.speed * 10.0);
                        
                        if (cmd->header.msg_type == "command") {
                            robot_controller_->waitForMotionComplete();
                        }
                    }
                }
            }
```

