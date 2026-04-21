#pragma once

#include <functional>

#include "camera/data/camera_types.hpp"

namespace camera {

/**
 * @brief 相机抽象接口。
 *
 * 约定：
 * - open(): 初始化/打开设备资源
 * - start(): 开始采集并通过回调输出帧
 * - close(): 停止采集并释放资源（可重复调用）
 */
class CameraInterface {
public:
    using FrameCallback = std::function<void(const CameraFrameView&)>;

    virtual ~CameraInterface() = default;

    /**
     * @brief 打开相机设备并完成必要初始化。
     * @return 成功返回 true，否则返回 false。
     */
    virtual bool open() = 0;

    /**
     * @brief 开始采集并触发帧回调。
     * @return 成功返回 true，否则返回 false。
     */
    virtual bool start() = 0;

    /**
     * @brief 停止采集并释放相关资源。
     */
    virtual void close() = 0;

    /**
     * @brief 设置帧回调。
     *
     * 相机实现应在采集线程/回调线程中调用该回调，回调内尽量避免长时间阻塞。
     */
    virtual void setFrameCallback(FrameCallback cb) = 0;
};

}
