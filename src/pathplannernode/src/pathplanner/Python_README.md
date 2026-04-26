# SamPathPlanning - 模块化重构版

## 📁 模块结构

```
SamPathPlanning/
├── attention_path.py          # 主入口文件，交互式流程控制
├── attention_computation.py   # 注意力热力图计算核心算法
├── pointcloud_processing.py   # 点云生成、法向量计算、查询
├── path_generation.py         # 路径生成算法（3 种方法）
├── path_optimization.py       # 路径插值、平滑、优化
├── local_coordinate.py        # 局部坐标系计算（3 种策略）
├── visualization.py           # 可视化功能封装
├── utils.py                   # 工具函数（深度预处理等）
└── README.md                  # 本说明文档
```

## 🔧 各模块功能详解

### 1. **attention_path.py** - 主入口与交互流程
**核心类**: `InteractiveSegmentation`

**功能**:
- 整合所有模块，提供完整的交互式分割→点云生成→路径规划流程
- 处理鼠标点击交互、SAM2 模型调用
- 协调各模块之间的数据传递

**主要方法**:
```python
__init__(depth_path)              # 初始化系统和各模块处理器
process_pipeline(color_path)      # 完整处理流程（主入口）
perform_segmentation()            # 执行 SAM2 分割
calculate_normals()               # 计算点云法向量
compute_local_frames()            # 计算局部坐标系
local_frames_to_scan_points()     # 转换为输出格式
```

---

### 2. **attention_computation.py** - 注意力热力图计算
**核心函数**: 
- `compute_internal_attention_from_masked_pc()` - 优化版注意力计算
- `visualize_internal_attention()` - 热力图可视化

**算法原理**:
1. 基于 k-d tree 的邻域搜索（支持半径/k 近邻）
2. 三大几何特征融合:
   - **法向量标准差**: 衡量表面弯曲程度
   - **PCA 线性度**: 识别棱边结构 (λ0/(λ0+λ1+λ2))
   - **深度 Laplacian**: 检测深度突变区域
3. 加权融合 → 归一化 → 高斯平滑

**参数示例**:
```python
attention_map, attention_values = compute_internal_attention_from_masked_pc(
    mask=mask,                    # 二值掩码
    pointcloud=points,            # 点云 (N,3)
    normals=normals,              # 法向量 (N,3)
    pixel_to_index=pixel2idx,     # 像素→点云索引映射
    k_neighbors=32,               # 邻域点数
    radius=0.015,                 # 邻域半径（米）
    weight_normal_std=0.4,        # 法向量权重
    weight_linearity=0.6,         # 线性度权重
    min_neighbors=6,              # 最小邻居数
    depth_map=depth               # 深度图（可选）
)
```

---

### 3. **pointcloud_processing.py** - 点云处理
**核心类**: `PointCloudProcessor`

**功能**:
- 深度图 + 掩码 → 彩色点云反投影
- 基于 Open3D 的法向量估计
- 像素坐标→3D 点快速查询（带邻域搜索容错）

**主要方法**:
```python
mask_depth_to_color_pointcloud(mask, depth_map, color_image)
    # 输入：掩码、深度图、彩色图
    # 输出：点云坐标 (N,3)、颜色 RGB(N,3)
    
calculate_normals(pointcloud, colors, radius=20, max_nn=50)
    # 使用 Open3D 估计法向量并定向
    
get_point_and_normal(u, v, search_radius=2)
    # 根据像素坐标查询 3D 点和法向量
    # 支持邻域搜索（当直接坐标无效时）
```

**相机模型**:
```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

---

### 4. **path_generation.py** - 路径生成
**核心类**: `PathGenerator`

**支持的 3 种路径生成算法**:

#### (1) **自适应轮廓栅格路径** (`generate_contour_adaptive_path`)
**原理**:
1. 计算掩码最小外接矩形（`cv2.minAreaRect`）
2. 确定长轴/短轴方向
3. 沿短轴方向生成等间距切割线
4. 计算切割线与轮廓的交点
5. 之字形连接交点形成路径

**特点**: 贴合不规则形状，边界覆盖好

#### (2) **旋转矩形网格路径** (`generate_rotated_rect_path`)
**原理**:
1. 基于外接矩形创建旋转坐标系
2. 在矩形内生成规则网格点
3. 过滤掉掩码外的点

**特点**: 路径规整，适合矩形/椭圆形区域

#### (3) **平滑轮廓跟随路径** (`generate_smooth_contour_path`)
**原理**:
1. 多边形逼近轮廓（`cv2.approxPolyDP`）
2. 沿多边形等弧长采样

**特点**: 路径连续光滑，适合曲线边界

**参数**:
```python
generator = PathGenerator(
    scan_mode='Short',    # 'Long'-沿长轴扫查 | 'Short'-沿短轴扫查
    spacing=10            # 切割线间距（像素）
)
contour, scan_points = generator.generate_from_mask(mask)
```

---

### 5. **path_optimization.py** - 路径优化
**核心类**: `PathOptimizer`

**功能**:

#### (1) **路径插值加密** (`interpolate_scan_points`)
- 奇偶段不同密度策略（偶数段默认 5 点，奇数段 30 点）
- 返回包含段信息的结构化数据

#### (2) **注意力引导插值** (`attention_guided_interpolation`)
- 根据热力图动态调整插值密度
- 高注意力区域 → 更密集（max_density）
- 低注意力区域 → 更稀疏（min_density）

```python
interpolated, info = optimizer.attention_guided_interpolation(
    scan_points, 
    attention_map,
    min_density=3,    # 最低密度
    max_density=15    # 最高密度
)
```

#### (3) **路径平滑** (`smooth_path_within_mask`)
- 滑动窗口平均（窗口大小可调）
- 强制约束点在掩码内

#### (4) **2D→3D 映射** (`map_2d_to_3d`)
- 将像素路径映射到点云空间
- 支持奇偶段点筛选策略

---

### 6. **local_coordinate.py** - 局部坐标系计算
**核心类**: `LocalCoordinateCalculator`

**目标**: 为每个路径点计算局部坐标系 `[x, y, z]`:
- **z 轴**: 法向量（已归一化）
- **x 轴**: 扫查方向（需满足 x⊥z）
- **y 轴**: 步进方向（y = z × x，右手系）

**3 种计算策略**:

#### (1) **统一方向法** (`method='uniform'`)
- 找到第一个偶数段的切线作为全局参考方向
- 所有点的 x 轴都使用该参考方向（投影到切平面）
- **适用场景**: 需要整体方向一致的路径

#### (2) **奇偶交替法** (`method='alternate'`)
- 奇数段：与第一个偶数段同向
- 偶数段：交替反向（段 0 正向，段 2 反向，段 4 正向...）
- **适用场景**: 往复式扫描，减少空行程

#### (3) **加权混合法** (`method='weighted'`, alpha=0.5)
- 局部切线方向与全局参考方向的加权混合
- `alpha=0`: 纯局部切线
- `alpha=1`: 纯全局方向
- **适用场景**: 平衡路径贴合度与方向一致性

**使用示例**:
```python
calculator = LocalCoordinateCalculator(method='uniform')
local_frames = calculator.compute(
    scan_points_3d,      # 3D 路径点
    scan_normals,        # 法向量
    scan_orig_indices    # 段索引（可选）
)

# 转换为输出格式 [x,y,z, x1,x2,x3, y1,y2,y3, nx,ny,nz]
output = calculator.frames_to_scan_points(local_frames)
```

---

### 7. **visualization.py** - 可视化
**核心类**: `PathVisualizer`

**功能**:
- `visualize_contour_path()`: 绘制轮廓和路径点（起点红/终点蓝）
- `visualize_segmentation_result()`: 叠加半透明绿色掩码
- `visualize_color_pointcloud()`: Open3D 交互式 3D 点云显示
- `draw_edge_on_mask()`: 边缘着色

---

### 8. **utils.py** - 工具函数
**核心函数**:

#### (1) **深度图预处理** (`preprocess_depth`)
- 最近邻填充空洞（基于 KDTree）
- 中值滤波去噪

#### (2) **旋转矩阵构造** (`rotation_matrix_from_z_to_v`)
- 构造旋转矩阵使 Z 轴对齐目标向量
- 用于坐标系变换

#### (3) **路径插值** (`interpolate_scan_points`)
- 基础线性插值功能

---

## 🚀 使用示例

### 完整流程
```python
from SamPathPlanning.attention_path import InteractiveSegmentation

# 1. 初始化
seg = InteractiveSegmentation("depth.png")

# 2. 加载图像并运行流程
seg.original_image = cv2.imread("color.png")
seg.process_pipeline("color.png")

# 3. 获取结果
scan_points_2d = seg.scan_points          # 2D 路径点
local_frames = seg.local_frames           # 局部坐标系
output_data = seg.local_frames_to_scan_points()  # 最终输出
```

### 单独使用某模块
```python
# 示例 1：仅使用路径生成
from SamPathPlanning.path_generation import PathGenerator

generator = PathGenerator(scan_mode='Short', spacing=10)
contour, path = generator.generate_from_mask(mask)

# 示例 2：仅计算注意力热力图
from SamPathPlanning.attention_computation import compute_internal_attention_from_masked_pc

attention_map, values = compute_internal_attention_from_masked_pc(
    mask, pointcloud, normals, pixel_to_index,
    weight_normal_std=0.4, weight_linearity=0.6
)

# 示例 3：自定义局部坐标系计算方法
from SamPathPlanning.local_coordinate import LocalCoordinateCalculator

calc_uniform = LocalCoordinateCalculator(method='uniform')
calc_alternate = LocalCoordinateCalculator(method='alternate')
calc_weighted = LocalCoordinateCalculator(method='weighted', alpha=0.7)
```

---

## 📊 数据流图

```
输入 (depth.png + color.png)
    ↓
[InteractiveSegmentation]
    ├─→ SAM2 分割 → current_mask
    ├─→ PointCloudProcessor → pointcloud + normals
    ├─→ PathGenerator → scan_points (2D)
    ├─→ PathOptimizer → interpolated_points
    ├─→ LocalCoordinateCalculator → local_frames
    └─→ PathVisualizer → 可视化结果
```

---

## ⚙️ 配置参数说明

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `default_shrink_factor` | 10 | 掩码腐蚀像素数（形态学收缩） |
| `default_InterPoins` | 5 | 偶数段插值点数 |
| `default_scan` | 'Short' | 扫查方向 ('Long'/'Short') |
| `default_spacing` | 10 | 切割线间距（像素） |
| `fx, fy` | ~500 | 相机焦距（像素） |
| `cx, cy` | ~(314,225) | 相机主点坐标 |

---

## 🔍 常见问题

### Q1: 如何选择局部坐标系计算方法？
- **uniform**: 需要所有路径点朝向一致（如单向扫查）
- **alternate**: 往复式扫描，减少机械臂空转
- **weighted**: 既要贴合路径又要方向稳定（调α参数）

### Q2: 注意力热力图不生效？
检查:
1. 点云是否有效生成（n_valid > 0）
2. 法向量是否计算成功
3. `weight_normal_std + weight_linearity = 1.0`
4. 尝试调整 `radius` 和 `k_neighbors`

### Q3: 路径点超出掩码范围？
启用平滑约束:
```python
optimizer = PathOptimizer()
smoothed = optimizer.smooth_path_within_mask(
    scan_points, mask_binary, window_size=5
)
```

---

## 📝 版本历史

**v2.0 (模块化重构版)**:
- ✅ 拆分为 8 个独立模块，职责清晰
- ✅ 移除冗余代码和注释掉的函数
- ✅ 统一命名规范和文档字符串
- ✅ 支持灵活组合各功能模块

**v1.0 (原始版)**:
- 单文件 2800+ 行，功能混杂

---

## 📧 联系方式

如有问题请提交 Issue 或联系开发团队。
