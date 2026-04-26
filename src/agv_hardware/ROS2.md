# 与server_node通信协议
---
## 结构化文本说明
### 消息格式
```json
{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "check"
    },
    "payload": {
    }
}
```
统一的 JSON 格式，包含 `header` 和 `payload` 两部分。	
- `header` 包含机器人 ID、类型和消息类别	
- `robot_id`：唯一标识每个机器人
	**vacuum_adsorption_robot**：气吸附机器人
	**Magnetic Adsorption Robot**：磁吸附机器人
	**mobile_dual_arm_robot**：AGV+双臂机器人
- `msg_type`：区分消息类型，定义如下：
	**Yes** : 机器人终端设备使能
	**command** : 控制命令
	**query** : 查询
- `payload` 包含具体的指令和参数，以及机器人状态信息。

例如：	
- `chassis`: 包含移动指令和参数(仅在msg_type为command时存在)	
- `arms`: 包含每个机械臂的指令和参数(仅在msg_type为command时存在)	
- `robot_info`: 包含机器人状态信息，如电池、电量、传感器数据等(在msg_type为status_update时存在)
## 命令示例
### 气吸附机器人(**vacuum_adsorption_robot**)
#### 设备使能
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "vacuum_adsorption_robot",
        "msg_type": "check"
    },
    "payload": {
        "radar": {
            "command": "check"
        },
        "camera": {
            "command": "check"
        },
        "agv": {
            "command": "check"
        }
    }
}


```
响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "vacuum_adsorption_robot",
        "msg_type": "check"
    },
    "payload": {
        "radar": {
            "command": "check",
            "result":"yes"
        },
        "camera": {
            "command": "check",
            "result":"yes"
        },
        "agv": {
            "command": "check",
            "result":"yes"
        }
    }
}
```

#### 单设备打开/关闭
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "vacuum_adsorption_robot",
        "msg_type": "check"
    },
    "payload": {
        "radar": {
            "command": "true"
        },
        "camera": {
            "command": "true"
        },
        "agv": {
            "command": "false"
        }
    }
}
```
响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "vacuum_adsorption_robot",
        "msg_type": "check"
    },
    "payload": {
        "radar": {
            "command": "true",
            "result":"yes"
        },
        "camera": {
            "command": "true",
            "result":"yes"
        },
        "agv": {
            "command": "false",
            "result":"yes"
        }
    }
}
```

#### AGV运动控制
##### 急停
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "emergency_stop"
        }
    }
}
```

响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "emergency_stop",
            "result":"yes"
        }
    }
}
```

##### 行驶
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "move",
            "data": {
                "vx": 0.0,
                "vy": 0.0,
                "wz": 0.0
            }
        }
    }
}
```
响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "move",
            "result":"yes"
        }
    }
}
```

#### 多通道超声采集/分析交互
##### 采集
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "capture"
        }
    }
}
```
响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "capture",
            "result":"capture_successed"
        }
    }
}
```
##### 保存
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "save"
        }
    }
}
```
响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "save",
            "result":"save_successed"
        }
    }
}
```
##### 作业完成
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "work_complete"
        }
    }
}
```
响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "agv",
        "msg_type": "command"
    },
    "payload": {
        "agv": {
            "command": "work_complete",
            "result":"work_complete_successed"
        }
    }
}
```

### AGV+双臂机器人(mobile_dual_arm_robot)
#### 设备使能

‘’‘
{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "check"
    },
    "payload": {
        "radar": {
            "command": "check"
        },
        "camera": {
            "command": "check"
        },
        "agv": {
            "command": "check"
        }
    }
}
’‘’

{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "response"
    },
    "payload": {
        "radar": {
            "command": "check",
            "result":"yes"
        },
        "camera": {
            "command": "check",
            "result":"yes"
        },
        "agv": {
            "command": "check",
            "result":"yes"
        }
    }
}
同 [[ROS2控制平台#设备使能]]

#### 单设备打开/关闭
同[[ROS2控制平台#单设备打开/关闭]]

---
#### 机械臂控制
##### 笛卡尔坐标系-机械臂控制
###### 直线规划
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "command"
    },
    "payload": {
        "arms": [
            {
                "arm_id": "left_arm",
                "command": "cartesian_move",
                "parameters": {
                    "x": 0.2,
                    "y": 0.1,
                    "z": 0.3,
                    "roll": 0.0,
                    "pitch": 0.5,
                    "yaw": 0.0,
                    "speed": 45
                }
            },
            {
                "arm_id": "right_arm",
                "command": "cartesian_move",
                "parameters": {
                    "x": -0.2,
                    "y": 0.1,
                    "z": 0.25,
                    "roll": 0.0,
                    "pitch": -0.5,
                    "yaw": 0.0,
                    "speed": 45
                }
            }
        ]
    }
}
```

响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "check"
    },
    "payload": {
        "arms": [
            {
                "arm_id": "left_arm",
                "command": "cartesian_move",
                "result": "cartesian_move_successed"
            },
            {
                "arm_id": "right_arm",
                "command": "cartesian_move",
                "result": "cartesian_move_successed"
            }
        ]
    }
}
```

###### 圆弧规划
请求：上位机 -> 下位机
```json

```

响应：下位机 -> 上位机
```json

```

##### 关节坐标系-机械臂控制
请求：上位机 -> 下位机
```json
{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "command"
    },
    "payload": {
        "arms": [
            {
                "arm_id": "left_arm",
                "command": "joint_move",
                "parameters": {
                    "val1": 0.2,
                    "val2": 0.1,
                    "val3": 0.3,
                    "val4": 0.0,
                    "val5": 0.5,
                    "val6": 0.0,
                    "speed": 45
                }
            },
            {
                "arm_id": "right_arm",
                "command": "joint_move",
                "parameters": {
                    "val1": -0.2,
                    "val2": 0.1,
                    "val3": 0.25,
                    "val4": 0.0,
                    "val5": -0.5,
                    "val6": 0.0,
                    "speed": 45
                }
            }
        ]
    }
}
```

响应：下位机 -> 上位机
```json
{
    "header": {
        "robot_id": "mobile_dual_arm_robot",
        "msg_type": "check"
    },
    "payload": {
        "arms": [
            {
                "arm_id": "left_arm",
                "command": "joint_move",
                "result": "joint_move_successed"
            },
            {
                "arm_id": "right_arm",
                "command": "joint_move",
                "result": "joint_move_successed"
            }
        ]
    }
}
```

##### 工件坐标系-机械臂控制
同[[ROS2控制平台#笛卡尔坐标系-机械臂控制]]

---
#### AGV控制
同 [[ROS2控制平台#行驶]]

--- 
## 视频流
``` json
{
  "header": {
    "robot_id": "AGV_ARM",
    "msg_type": "status_sensor"
  },
  "payload": {
    "camera_url": "rtsp://192.168.1.100:554/live"
  }
}
```
---
## 点云数据
```json

```

---
# robot.js(临时文件)
```json
{
    "header": {
        "robot_id": "AGV_ARM",
        "msg_type": "command"
    },
    "payload": {
        "chassis": {
            "command": "move",
            "parameters": {
                "x": 0.1,
                "y": 0.0,
                "yaw": 0.8,
                "speed": 0.5
            }
        },
        "arms": [
            {
                "arm_id": "left_arm",
                "command": "cartesian_move",
                "parameters": {
                    "x": 0.2,
                    "y": 0.1,
                    "z": 0.3,
                    "roll": 0.0,
                    "pitch": 0.5,
                    "yaw": 0.0
                }
            },
            {
                "arm_id": "right_arm",
                "command": "cartesian_move",
                "parameters": {
                    "x": -0.2,
                    "y": 0.1,
                    "z": 0.25,
                    "roll": 0.0,
                    "pitch": -0.5,
                    "yaw": 0.0
                }
            }
        ],
        "robot_info": {
            "battery": true,
            "lidar": true,
            "camera": true,
            "arm1_position": true,
            "arm2_position": true,
            "agv_position": true
        }
    }
}
```

