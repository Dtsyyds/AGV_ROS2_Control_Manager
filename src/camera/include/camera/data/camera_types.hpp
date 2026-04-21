#pragma once

#include <cstddef>
#include <string>

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

}
