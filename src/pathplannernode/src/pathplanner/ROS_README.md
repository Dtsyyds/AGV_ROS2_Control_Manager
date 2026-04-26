# PathPlanner ROS2 节点说明

本文档详细说明 `pathplanner_ros` 包中的 ROS2 节点功能和使用方法。

---

## 目录

1. [节点概览](#节点概览)
2. [pathplanner_ros2_node](#pathplanner_ros2_node)
3. [test_click_publisher](#test_click_publisher)
4. [启动方式](#启动方式)
5. [坐标系说明](#坐标系说明)
6. [调试与测试](#调试与测试)

---

## 节点概览

| 节点名 | 文件 | 功能描述 |
|--------|------|----------|
| `pathplanner_ros2_node` | `pathplanner_ros2_node.py` | 主路径规划节点，基于 SAM2 分割和注意力机制生成超声扫描路径 |
| `test_click_publisher` | `test_click_publisher.py` | 测试节点，发布点击点触发路径规划 |

---

## pathplanner_ros2_node

### 功能描述

基于 SAM2 分割和注意力机制的超声扫描路径规划节点，支持两种触发模式：

- **话题触发模式**: 接收 `/click_point` 话题的像素坐标触发路径规划
- **交互模式**: 通过 OpenCV 弹窗手动选择点击点（设置 `interactive_mode:=true`）

同时，节点会发布相机到机械臂的 TF 变换（`arm1_tool_link` → `camera_link`）。

### 订阅话题

| 话题名 | 类型 | 默认话题 | 描述 |
|--------|------|----------|------|
| 彩色图像 | `sensor_msgs/Image` | `color/image_raw` | 彩色图像 (BGR8) |
| 深度图像 | `sensor_msgs/Image` | `depth/image_raw` | 深度图像 (16UC1 或 32FC1) |
| 点击点 | `geometry_msgs/PointStamped` | `/click_point` | 点击点像素坐标 (x, y) |

### 发布话题

| 话题名 | 类型 | 默认话题 | 描述 |
|--------|------|----------|------|
| 3D 路径 | `sensor_msgs/PointCloud2` | `/path_planner/path_3d` | 3D 路径点 [x,y,z,rx,ry,rz]（单位：mm + 弧度） |
| 笛卡尔路径 | `geometry_msgs/PoseArray` | `/path_planner/cartesian_path` | 笛卡尔路径（位置单位 m，姿态为四元数） |

### TF 发布

- **父坐标系**: `arm1_tool_link`（机械臂末端关节）
- **子坐标系**: `camera_link`（相机）
- **发布频率**: 10 Hz
- **用途**: 手眼标定，将相机坐标系下的路径点转换到机械臂基坐标系

### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `color_topic` | string | `color/image_raw` | 彩色图像话题名 |
| `depth_topic` | string | `depth/image_raw` | 深度图像话题名 |
| `click_point_topic` | string | `/click_point` | 点击点话题名 |
| `output_3d_path_topic` | string | `/path_planner/path_3d` | 3D 路径输出话题名 |
| `output_cartesian_path_topic` | string | `/path_planner/cartesian_path` | 笛卡尔路径输出话题名 |
| `scan_mode` | string | `Long` | 扫描模式: `Long` / `Short` |
| `spacing` | int | `10` | 步进间隔（像素） |
| `shrink_factor` | int | `12` | 掩码收缩因子（像素） |
| `fx` | float | `498.3686770748583` | 相机内参 fx |
| `fy` | float | `501.9355502582987` | 相机内参 fy |
| `cx` | float | `314.3019441792476` | 相机内参 cx（主点 x） |
| `cy` | float | `225.6695918834769` | 相机内参 cy（主点 y） |
| `model_path` | string | `''` | SAM2 模型路径 |
| `interactive_mode` | bool | `false` | 是否启用交互模式（OpenCV 弹窗） |
| `enable_visualization` | bool | `true` | 是否启用可视化 |

---

## test_click_publisher

### 功能描述

测试节点，用于发布点击点来触发路径规划。适用于测试 `pathplanner_ros2_node` 的功能。

### 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `x` | int | `320` | 点击点 x 坐标（像素） |
| `y` | int | `240` | 点击点 y 坐标（像素） |
| `topic` | string | `/click_point` | 发布的话题名 |

### 使用示例

```bash
# 发布默认点击点 (320, 240)
ros2 run pathplanner test_click_publisher

# 发布指定点击点 (400, 300)
ros2 run pathplanner test_click_publisher --ros-args -p x:=400 -p y:=300
```

---

## 启动方式

### 方式 1: 使用 `ros2 run` 启动主节点

#### 基本启动（使用默认参数）

```bash
ros2 run pathplanner pathplanner_ros2_node
```

#### 带参数启动（非交互模式，推荐用于实际运行）

```bash
ros2 run pathplanner pathplanner_ros2_node \
  --ros-args \
  -p color_topic:=/camera/color/image_raw \
  -p depth_topic:=/camera/depth/image_raw \
  -p scan_mode:=Short \
  -p spacing:=15 \
  -p shrink_factor:=10 \
  -p interactive_mode:=false
```

#### 交互模式启动（推荐用于调试）

```bash
ros2 run pathplanner pathplanner_ros2_node \
  --ros-args \
  -p interactive_mode:=true \
  -p enable_visualization:=true
```

### 方式 2: 使用启动文件

```bash
# 如果有 launch 文件
ros2 launch pathplanner pathplanner_launch.py
```

### 方式 3: 使用参数文件

创建 `params.yaml` 文件：

```yaml
pathplanner_ros2_node:
  ros__parameters:
    color_topic: "/camera/color/image_raw"
    depth_topic: "/camera/depth/image_raw"
    scan_mode: "Short"
    spacing: 15
    shrink_factor: 10
    interactive_mode: false
    enable_visualization: true
```

使用参数文件运行：

```bash
ros2 run pathplanner pathplanner_ros2_node --ros-args --params-file params.yaml
```

---

## 坐标系说明

### 输入坐标系

| 数据 | 坐标系 | 单位 |
|------|--------|------|
| 点击点 (x, y) | 图像坐标系 | 像素 (px) |
| 深度值 | 相机坐标系 | 毫米 (mm) |

### 输出坐标系

| 话题 | 坐标系 | 位置单位 | 姿态表示 |
|------|--------|----------|----------|
| `/path_planner/path_3d` | `camera_link` | 毫米 (mm) | 欧拉角 (rx, ry, rz)，单位：弧度 |
| `/path_planner/cartesian_path` | `camera_link` | 米 (m) | 四元数 (x, y, z, w) |

### 坐标转换流程

```
图像坐标 (u, v) + 深度值
        ↓
相机内参反投影 → 相机坐标系 3D 点 (mm)
        ↓
路径规划算法 → 生成扫描路径点
        ↓
发布到 ROS 话题
```

### TF 变换关系

```
arm1_tool_link (机械臂末端)
        │
        │ TF: 手眼标定变换
        ▼
camera_link (相机)
        │
        │ 路径点在此坐标系下
        ▼
路径点 (x, y, z, rx, ry, rz)
```

**TF 变换参数**（在代码中硬编码）：

- **平移**: `[-23.88, -215.28, 141.76]` mm
- **旋转（四元数）**: `[-0.747, 0.001, -0.006, 0.665]` (x, y, z, w)

---

## 调试与测试

### 典型使用流程

#### 场景 1: 话题触发模式（推荐用于实际运行）

```bash
# 终端 1: 启动路径规划节点
ros2 run pathplanner pathplanner_ros2_node

# 终端 2: 发布点击点触发路径规划
ros2 run pathplanner test_click_publisher --ros-args -p x:=320 -p y:=240

# 终端 3: 查看输出的笛卡尔路径
ros2 topic echo /path_planner/cartesian_path
```

#### 场景 2: 交互模式（推荐用于调试）

```bash
# 启动交互模式
ros2 run pathplanner pathplanner_ros2_node --ros-args -p interactive_mode:=true
```

然后会弹出 OpenCV 窗口，在图像上点击选择目标区域。

### 查看输出

```bash
# 查看 3D 路径 (PointCloud2)
ros2 topic echo /path_planner/path_3d

# 查看笛卡尔路径 (PoseArray)
ros2 topic echo /path_planner/cartesian_path

# 查看话题列表[1776684434.040963316] [pathplanner_ros2_node]: 订阅彩色图: color/image_raw

ros2 topic list

# 查看节点图
rqt_graph
```

### RViz2 可视化

```bash
# 启动 RViz2
rviz2
```

在 RViz2 中添加以下显示：
- **TF**: 显示坐标系关系
- **PoseArray**: 订阅 `/path_planner/cartesian_path`，显示路径点姿态
- **PointCloud2**: 订阅 `/path_planner/path_3d`，显示 3D 路径点

### 常用调试命令

```bash
# 查看节点信息
ros2 node info /pathplanner_ros2_node

# 查看话题信息
ros2 topic info /path_planner/cartesian_path

# 查看 TF 树
ros2 run tf2_tools view_frames

# 实时查看 TF 变换
ros2 run tf2_ros tf2_echo arm1_tool_link camera_link
```

---

## 注意事项

1. **模型依赖**: 节点需要 SAM2 模型支持，确保 `model_path` 参数指向正确的模型文件路径

2. **相机内参**: 默认内参针对 RealSense D435 相机，如果使用其他相机需要修改 `fx`, `fy`, `cx`, `cy` 参数

3. **TF 标定**: 手眼标定参数（`arm1_tool_link` → `camera_link`）在代码中硬编码，如果相机安装位置改变需要更新代码

4. **深度图格式**: 确保深度图像的单位正确（通常为毫米），且与彩色图像对齐

5. **坐标系方向**: 
   - 图像坐标系: 原点在左上角，x 向右，y 向下
   - 相机坐标系: z 向前，x 向右，y 向下（符合 ROS 相机坐标系约定）
