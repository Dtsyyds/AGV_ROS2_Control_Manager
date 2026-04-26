










#include "camera/astras/astras_camera.hpp"

#include <cstring>
#include <utility>

#include <rclcpp/rclcpp.hpp>
#include <OniCTypes.h>
#include <PS1080.h>

namespace camera {

/**
 * @brief AstraS 相机实现（OpenNI2）。
 *
 * 注意：OpenNI 的初始化/反初始化由 open()/close() 管理；析构时会调用 close() 释放资源。
 */
AstraSCamera::AstraSCamera(bool enable_color, bool enable_depth, bool enable_ir)
    : enable_color_(enable_color),
      enable_depth_(enable_depth),
      enable_ir_(enable_ir) {}

/**
 * @brief 析构时确保停止流并关闭设备。
 */
AstraSCamera::~AstraSCamera() {
    close();
}

/**
 * @brief 设置帧回调。
 */
void AstraSCamera::setFrameCallback(FrameCallback cb) {
    frame_cb_ = std::move(cb);
}

/**
 * @brief 初始化 OpenNI2 并打开任意可用设备。
 *
 * 失败时将扩展错误保存到 last_error_，并返回 false。
 */
bool AstraSCamera::open() { 
    openni::Status rc = openni::OpenNI::initialize();
    if (rc != openni::STATUS_OK) {
        last_error_ = openni::OpenNI::getExtendedError();
        return false;
    }

    rc = device_.open(openni::ANY_DEVICE);
    if (rc != openni::STATUS_OK) {
        last_error_ = openni::OpenNI::getExtendedError();
        return false;
    }

    return true;
}

/**
 * @brief 根据启用开关创建/配置并启动各视频流。
 *
 * 同时启用彩色与深度时，如果设备支持则打开深度到彩色的配准模式。
 */
bool AstraSCamera::start() {
    if (enable_color_) {
        if (!setupStream(color_stream_, openni::SENSOR_COLOR, openni::PIXEL_FORMAT_RGB888)) {
            return false;
        }
    }

    if (enable_depth_) {
        if (!setupStream(depth_stream_, openni::SENSOR_DEPTH, openni::PIXEL_FORMAT_DEPTH_1_MM)) {
            return false;
        }
    }

    if (enable_ir_) {
        if (!setupStream(ir_stream_, openni::SENSOR_IR, openni::PIXEL_FORMAT_GRAY16)) {
            return false;
        }
    }

    if (enable_color_ && enable_depth_) {
        if (device_.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR)) {
            device_.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);
        }
    }

    return true;
}

/**
 * @brief 停止流、关闭设备并 shutdown OpenNI。
 *
 * 允许重复调用，内部会检查资源有效性。
 */
void AstraSCamera::close() {
    stopStream(color_stream_, color_listener_.get());
    stopStream(depth_stream_, depth_listener_.get());
    stopStream(ir_stream_, ir_listener_.get());

    if (device_.isValid()) {
        device_.close();
    }

    openni::OpenNI::shutdown();
}

const std::string& AstraSCamera::lastError() const {
    return last_error_;
}

/**
 * @brief OpenNI 新帧回调：读取 VideoFrameRef 并封装为 CameraFrameView 转发给上层。
 */
AstraSCamera::StreamListener::StreamListener(AstraSCamera& camera,
                                             CameraStreamType type,
                                             std::string frame_id,
                                             std::string encoding)
    : camera_(camera),
      type_(type),
      frame_id_(std::move(frame_id)),
      encoding_(std::move(encoding)) {}

void AstraSCamera::StreamListener::onNewFrame(openni::VideoStream& stream) {
    if (!camera_.frame_cb_) {
        return;
    }

    openni::VideoFrameRef frame;
    if (stream.readFrame(&frame) != openni::STATUS_OK) {
        return;
    }

    CameraFrameView view;
    view.type = type_;
    view.width = frame.getWidth();
    view.height = frame.getHeight();
    view.encoding = encoding_;
    view.frame_id = frame_id_;
    view.data = frame.getData();
    view.data_size = static_cast<std::size_t>(frame.getDataSize());

    camera_.frame_cb_(view);
}

/**
 * @brief 创建并启动指定传感器类型的视频流。
 *
 * - 固定配置为 1280x960@30fps
 * - 根据传感器类型设置 frame_id 与 encoding
 * - 注册 StreamListener，将帧通过 FrameCallback 输出
 */
bool AstraSCamera::setupStream(openni::VideoStream& stream,
                               openni::SensorType type,
                               openni::PixelFormat pixel_format) {
    if (device_.getSensorInfo(type) == nullptr) {
        return false;
    }

    openni::Status rc = stream.create(device_, type);
    if (rc != openni::STATUS_OK) {
        last_error_ = openni::OpenNI::getExtendedError();
        return false;
    }

    openni::VideoMode mode;
    mode.setResolution(640, 480);
    mode.setFps(30);
    mode.setPixelFormat(pixel_format);
    stream.setVideoMode(mode);

    CameraStreamType stream_type = CameraStreamType::Color;
    std::string frame_id;
    std::string encoding;

    if (type == openni::SENSOR_COLOR) {
        stream_type = CameraStreamType::Color;
        frame_id = "camera_color_frame";
        encoding = "rgb8";
    } else if (type == openni::SENSOR_DEPTH) {
        stream_type = CameraStreamType::Depth;
        frame_id = "camera_depth_frame";
        encoding = "16UC1";
    } else if (type == openni::SENSOR_IR) {
        stream_type = CameraStreamType::Ir;
        frame_id = "camera_ir_frame";
        encoding = "mono16";
    }

    auto listener = std::make_unique<StreamListener>(*this, stream_type, frame_id, encoding);
    stream.addNewFrameListener(listener.get());

    rc = stream.start();
    if (rc != openni::STATUS_OK) {
        stream.removeNewFrameListener(listener.get());
        last_error_ = openni::OpenNI::getExtendedError();
        return false;
    }

    if (type == openni::SENSOR_COLOR) {
        color_listener_ = std::move(listener);
    } else if (type == openni::SENSOR_DEPTH) {
        depth_listener_ = std::move(listener);
    } else if (type == openni::SENSOR_IR) {
        ir_listener_ = std::move(listener);
    }

    return true;
}

/**
 * @brief 停止视频流并注销监听器，释放 OpenNI stream 资源。
 */
void AstraSCamera::stopStream(openni::VideoStream& stream, openni::VideoStream::NewFrameListener* listener) {
    if (listener != nullptr) {
        stream.removeNewFrameListener(listener);
    }

    if (stream.isValid()) {
        stream.stop();
        stream.destroy();
    }
}

/**
 * @brief 从设备获取相机参数结构体。
 *
 * 使用 SDK 定义的 OBCameraParamsData 结构体（定义在 PS1080.h）：
 * - depthRes: 深度分辨率
 * - colorRes: 彩色分辨率
 * - params: OBCameraParams 内参数据
 *   - l_intr_p[4]: 左相机/深度相机内参 [fx, fy, cx, cy]
 *   - r_intr_p[4]: 右相机/彩色相机内参 [fx, fy, cx, cy]
 *   - l_k[5]: 左相机畸变系数 [k1, k2, p1, p2, k3]
 *   - r_k[5]: 右相机畸变系数 [k1, k2, p1, p2, k3]
 *
 * 注意：需要使用 OBEXTENSION_ID_CAM_PARAMS (14) 获取参数
 */
static std::optional<OBCameraParams> getCameraParams(openni::Device& device) {
    // 尝试直接使用 OBCameraParams，不使用包装结构体
    OBCameraParams params;
    std::memset(&params, 0, sizeof(params));
    int data_size = sizeof(params);

    openni::Status rc = device.getProperty(openni::OBEXTENSION_ID_CAM_PARAMS, &params, &data_size);
    if (rc != openni::STATUS_OK) {
        return std::nullopt;
    }

    return params;
}

/**
 * @brief 获取彩色相机的内参。
 */
std::optional<CameraIntrinsics> AstraSCamera::getColorIntrinsics() {
    if (!device_.isValid()) {
        return std::nullopt;
    }

    auto params_opt = getCameraParams(device_);
    if (!params_opt.has_value()) {
        return std::nullopt;
    }

    const OBCameraParams& params = params_opt.value();
    CameraIntrinsics intrinsics;
    // Color 内参使用 r_intr_p: [fx, fy, cx, cy]
    intrinsics.fx = static_cast<double>(params.r_intr_p[0]);
    intrinsics.fy = static_cast<double>(params.r_intr_p[1]);
    intrinsics.cx = static_cast<double>(params.r_intr_p[2]);
    intrinsics.cy = static_cast<double>(params.r_intr_p[3]);
    intrinsics.distortion_coeffs = {static_cast<double>(params.l_k[0]),
                                    static_cast<double>(params.l_k[1]),
                                    static_cast<double>(params.l_k[2]),
                                    static_cast<double>(params.l_k[3]),
                                    static_cast<double>(params.l_k[4])};
    return intrinsics;
}

/**
 * @brief 获取深度相机的内参。
 */
std::optional<CameraIntrinsics> AstraSCamera::getDepthIntrinsics() {
    if (!device_.isValid()) {
        return std::nullopt;
    }

    auto params_opt = getCameraParams(device_);
    if (!params_opt.has_value()) {
        return std::nullopt;
    }

    const OBCameraParams& params = params_opt.value();
    CameraIntrinsics intrinsics;
    // Depth 内参使用 l_intr_p: [fx, fy, cx, cy]
    intrinsics.fx = static_cast<double>(params.l_intr_p[0]);
    intrinsics.fy = static_cast<double>(params.l_intr_p[1]);
    intrinsics.cx = static_cast<double>(params.l_intr_p[2]);
    intrinsics.cy = static_cast<double>(params.l_intr_p[3]);
    intrinsics.distortion_coeffs = {static_cast<double>(params.r_k[0]),
                                    static_cast<double>(params.r_k[1]),
                                    static_cast<double>(params.r_k[2]),
                                    static_cast<double>(params.r_k[3]),
                                    static_cast<double>(params.r_k[4])};
    return intrinsics;
}

} 
