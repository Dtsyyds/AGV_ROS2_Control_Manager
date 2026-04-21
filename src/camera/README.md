针对Astra-s
一、下载sdk
https://www.orbbec.com/developers/openni-sdk/
zyj@zyj-Lenovo:~/下载$ ls
 OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux
 OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux_x64.zip
 OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d_arm64.zip
'Orbbec-OpenNI2.0 SDK C++接口使用规范-1.0.2.pdf'
 Orbbec_OpenNI_v2.3.0.86-beta6_linux_release.zip

根据平台选择
ros2 需要sdk文件夹


zyj@zyj-Lenovo:~/下载/OpenNI_2.3.0.86_202210111154_4c8f5aa4_beta6_linux$ tree -L 4
.
├── rules
│   ├── install.sh
│   └── orbbec-usb.rules
└── sdk
    ├── Include
    │   ├── Android-Arm
    │   │   └── OniPlatformAndroid-Arm.h
    │   ├── Driver
    │   │   ├── OniDriverAPI.h
    │   │   └── OniDriverTypes.h
    │   ├── KinectProperties.h
    │   ├── Linux-Arm
    │   │   └── OniPlatformLinux-Arm.h
    │   ├── Linux-x86
    │   │   └── OniPlatformLinux-x86.h
    │   ├── MacOSX
    │   │   └── OniPlatformMacOSX.h
    │   ├── OniCAPI.h
    │   ├── OniCEnums.h
    │   ├── OniCProperties.h
    │   ├── OniCTypes.h
    │   ├── OniEnums.h
    │   ├── OniPlatform.h
    │   ├── OniProperties.h
    │   ├── OniTest.h
    │   ├── OniVersion.h
    │   ├── OpenNI.h
    │   ├── PrimeSense.h
    │   ├── PS1080.h
    │   ├── PSLink.h
    │   ├── SonixCamera.h
    │   ├── util.h
    │   └── Win32
    │       └── OniPlatformWin32.h
    ├── libs
    │   ├── libOpenNI2.so
    │   ├── OpenNI2
    │   │   └── Drivers
    │   └── OpenNI.ini
    └── thirdpart
        ├── Libusb-1.0.9 Installation instructions.pdf
        └── libusb-1.0.9.tar.gz

二、构建包
zyj@zyj-Lenovo:~/Code/ROS2/new_agv_robot/agv_ros2/src$ ros2 pkg create camera --build-type ament_cmake --dependencies rclcpp sensor_msgs cv_bridge image_transport

复制sdk目录到相机包中
zyj@zyj-Lenovo:~/Code/ROS2/new_agv_robot/agv_ros2/src/camera$ tree -L 4
.
├── CMakeLists.txt
├── include
│   └── camera
├── package.xml
├── sdk
│   ├── Include
│   │   ├── Android-Arm
│   │   │   └── OniPlatformAndroid-Arm.h
│   │   ├── Driver
│   │   │   ├── OniDriverAPI.h
│   │   │   └── OniDriverTypes.h
│   │   ├── KinectProperties.h
│   │   ├── Linux-Arm
│   │   │   └── OniPlatformLinux-Arm.h
│   │   ├── Linux-x86
│   │   │   └── OniPlatformLinux-x86.h
│   │   ├── MacOSX
│   │   │   └── OniPlatformMacOSX.h
│   │   ├── OniCAPI.h
│   │   ├── OniCEnums.h
│   │   ├── OniCProperties.h
│   │   ├── OniCTypes.h
│   │   ├── OniEnums.h
│   │   ├── OniPlatform.h
│   │   ├── OniProperties.h
│   │   ├── OniTest.h
│   │   ├── OniVersion.h
│   │   ├── OpenNI.h
│   │   ├── PrimeSense.h
│   │   ├── PS1080.h
│   │   ├── PSLink.h
│   │   ├── SonixCamera.h
│   │   ├── util.h
│   │   └── Win32
│   │       └── OniPlatformWin32.h
│   ├── libs
│   │   ├── libOpenNI2.so
│   │   ├── OpenNI2
│   │   │   └── Drivers
│   │   └── OpenNI.ini
│   └── thirdpart
│       ├── Libusb-1.0.9 Installation instructions.pdf
│       └── libusb-1.0.9.tar.gz
└── src

三、添加相机数据接口camera_types.hpp
#pragma once

#include <cstddef>
#include <string>

namespace camera {

enum class CameraStreamType {
    Color,
    Depth,
    Ir,
};

struct CameraFrameView {
    CameraStreamType type;
    int width;
    int height;
    std::string encoding;
    std::string frame_id;
    const void* data;
    std::size_t data_size;
};

}

四、抽象父类camera_interface.hpp
#pragma once

#include <functional>

#include "camera/camera_types.hpp"

namespace camera {

class CameraInterface {
public:
    using FrameCallback = std::function<void(const CameraFrameView&)>;

    virtual ~CameraInterface() = default;
    virtual bool open() = 0;
    virtual bool start() = 0;
    virtual void close() = 0;
    virtual void setFrameCallback(FrameCallback cb) = 0;
};

}


五、子类AstraS_camera.hpp
#pragma once

#include <memory>
#include <string>

#include <OpenNI.h>

#include "camera/camera_interface.hpp"

namespace camera {

class AstraSCamera final : public CameraInterface {
public:
    AstraSCamera(bool enable_color, bool enable_depth, bool enable_ir);
    ~AstraSCamera() override;

    void setFrameCallback(FrameCallback cb) override;
    bool open() override;
    bool start() override;
    void close() override;

    const std::string& lastError() const;

private:
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

    bool setupStream(openni::VideoStream& stream, openni::SensorType type, openni::PixelFormat pixel_format);
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

