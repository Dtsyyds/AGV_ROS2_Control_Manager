#pragma once

#include <memory>
#include <string>

#include <OpenNI.h>

#include "camera/interface/camera_interface.hpp"

namespace camera {

/**
 * @brief Orbbec AstraS 相机实现（基于 OpenNI2）。
 *
 * 负责打开设备、配置并启动 Color/Depth/IR 视频流，通过 FrameCallback 输出帧。
 */
class AstraSCamera final : public CameraInterface {
public:
    /**
     * @brief 构造函数。
     * @param enable_color 是否启用彩色流
     * @param enable_depth 是否启用深度流
     * @param enable_ir 是否启用红外流
     */
    AstraSCamera(bool enable_color, bool enable_depth, bool enable_ir);
    ~AstraSCamera() override;

    /**
     * @brief 设置帧回调。
     */
    void setFrameCallback(FrameCallback cb) override;

    /**
     * @brief 初始化 OpenNI2 并打开设备。
     */
    bool open() override;

    /**
     * @brief 启动已启用的视频流。
     */
    bool start() override;

    /**
     * @brief 停止视频流并关闭设备。
     */
    void close() override;

    /**
     * @brief 获取最近一次错误信息。
     */
    const std::string& lastError() const;

private:
    /**
     * @brief OpenNI2 新帧监听器，将 SDK 帧转换为 CameraFrameView 并透传给上层回调。
     */
    class StreamListener final : public openni::VideoStream::NewFrameListener {
    public:
        StreamListener(AstraSCamera& camera,
                       CameraStreamType type,
                       std::string frame_id,
                       std::string encoding);

        void onNewFrame(openni::VideoStream& stream) override;

    private:
        AstraSCamera& camera_;
        CameraStreamType type_;
        std::string frame_id_;
        std::string encoding_;
    };

    /**
     * @brief 配置并创建指定类型的视频流。
     */
    bool setupStream(openni::VideoStream& stream, openni::SensorType type, openni::PixelFormat pixel_format);

    /**
     * @brief 停止并注销监听器。
     */
    void stopStream(openni::VideoStream& stream, openni::VideoStream::NewFrameListener* listener);

    bool enable_color_;
    bool enable_depth_;
    bool enable_ir_;

    FrameCallback frame_cb_;
    std::string last_error_;

    openni::Device device_;
    openni::VideoStream depth_stream_;
    openni::VideoStream ir_stream_;
    openni::VideoStream color_stream_;

    std::unique_ptr<StreamListener> color_listener_;
    std::unique_ptr<StreamListener> depth_listener_;
    std::unique_ptr<StreamListener> ir_listener_;
};

}
