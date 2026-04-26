#pragma once

#include <cstddef>
#include <string>
#include <array>

namespace camera {

/**
 * @brief 相机数据流类型。
 *
 * 用于区分彩色、深度、红外等不同输出流。
 */
enum class CameraStreamType {
    Color,
    Depth,
    Ir,
};

/**
 * @brief 相机帧的只读视图。
 *
 * data 指向帧数据的外部存储（由相机/SDK 管理），回调返回后该指针可能失效；
 * 上层如需异步使用，请自行拷贝。
 */
struct CameraFrameView {
    CameraStreamType type;  ///< 数据流类型
    int width;              ///< 图像宽度（像素）
    int height;             ///< 图像高度（像素）
    std::string encoding;   ///< 编码/像素格式标识（与上层约定）
    std::string frame_id;   ///< ROS frame_id
    const void* data;       ///< 帧数据指针（只读）
    std::size_t data_size;  ///< data 字节数
};

/**
 * @brief 相机内参结构体。
 *
 * 包含焦距 (fx, fy)、主点 (cx, cy) 和畸变系数 (k1, k2, p1, p2, k3)。
 * 用于填充 sensor_msgs/CameraInfo。
 */
struct CameraIntrinsics {
    double fx;  ///< X 轴焦距（像素）
    double fy;  ///< Y 轴焦距（像素）
    double cx;  ///< X 轴主点（像素）
    double cy;  ///< Y 轴主点（像素）
    std::array<double, 5> distortion_coeffs;  ///< 畸变系数 [k1, k2, p1, p2, k3]

    /**
     * @brief 检查内参是否有效。
     * @return 如果 fx > 0 且 fy > 0 返回 true，否则返回 false。
     */
    bool isValid() const {
        return fx > 0.0 && fy > 0.0;
    }
};

}
