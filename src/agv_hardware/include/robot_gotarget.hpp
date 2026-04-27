#ifndef ROBOT_DRIVER_HPP
#define ROBOT_DRIVER_HPP
#include <iostream>
#include <fcntl.h> /*文件控制定义*/
#include <rclcpp/rclcpp.hpp>

// 检查是否已经通过 termios.h 定义了 struct termios
// 如果是，则不再包含 asm/termbits.h 以避免冲突
// #if !defined(_TERMBITS_H) && !defined(_BITS_TERMIOS_H)
// #include <asm/termbits.h> // 支持非标准波特率
// #else
// #include <termios.h> // 使用标准 termios
// #endif

#include <sys/ioctl.h>
#include <vector>
#include <string>
#include <cstring>
#include "cJSON.h"
// Linux Socket & Serial headers
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

// using namespace std;

#define SEERCONTROLIP "10.42.0.114"//AGV IP
// ==========================================
// 1. 抽象基类 (Base Class)
// ==========================================
class RobotMove 
{
private:

public:
    RobotMove() = default;

    virtual bool init(const std::string& connection_str) = 0;//初始化硬件连接（TCP/SBUS）

    virtual void setVelocity(double vx, double vy, double wz) = 0;

    virtual void send_loop() = 0;

    virtual void moveshutdown() = 0;

    /* -----------------仅气吸附和涵道控制方法-----------------*/
    virtual void setScannerConfig(double speed, double distance, double precision){};

    virtual void scannercontrol(){};

    /* -----------------AGV 独有控制方法-----------------*/
    // 特殊动作接口 (仅 AGV 支持, 气吸附返回 false)
    virtual bool movebydistance(double dist, double vx, double vy) { return false; }
    virtual bool rotatebyangle(double angle, double vw) { return false; }
    // 获取位置接口 (x, y, theta)
    virtual bool getpose(double& x, double& y, double& theta) { return false; }



    // virtual ~RobotMove() = default;
};

// ==========================================
// 2. 气吸附机器人底盘控制实现 (UART/SBUS)
// ==========================================
class AirRobotMove : public RobotMove
{
private:
    int serial_fd_ = -1;//串口连接标志位
    std::vector<uint16_t> ch_;//通道数据
    // 核心打包函数：16通道 -> 25字节,1个通道11bit
    void pack_protocol_data(std::vector<uint16_t> ch, uint8_t* buf) 
    {
        if (ch.size() != 16) 
        {
            std::cerr << "需要 16 个通道数据" << std::endl;
            return ;
        }

        for (int i = 0; i < 16; ++i) 
        {
            if(ch[i] > 2047)
            {
                ch[i] = 2047;
            }
        }

        buf[0] = 0x0F; // 帧头

        // 前 8 通道
        buf[1]  = (uint8_t)((ch[0] >> 3) & 0xFF);
        buf[2]  = (uint8_t)(((ch[0] << 5) | (ch[1] >> 6)) & 0xFF);
        buf[3]  = (uint8_t)(((ch[1] << 2) | (ch[2] >> 9)) & 0xFF);
        buf[4]  = (uint8_t)((ch[2] >> 1) & 0xFF);
        buf[5]  = (uint8_t)(((ch[2] << 7) | (ch[3] >> 4)) & 0xFF);
        buf[6]  = (uint8_t)(((ch[3] << 4) | (ch[4] >> 7)) & 0xFF);
        buf[7]  = (uint8_t)(((ch[4] << 1) | (ch[5] >> 10)) & 0xFF);
        buf[8]  = (uint8_t)((ch[5] >> 2) & 0xFF);
        buf[9]  = (uint8_t)(((ch[5] << 6) | (ch[6] >> 5)) & 0xFF);
        buf[10] = (uint8_t)(((ch[6] << 3) | (ch[7] >> 8)) & 0xFF);
        buf[11] = (uint8_t)(ch[7] & 0xFF);

        // 后 8 通道
        buf[12] = (uint8_t)((ch[8] >> 3) & 0xFF);
        buf[13] = (uint8_t)(((ch[8] << 5) | (ch[9] >> 6)) & 0xFF);
        buf[14] = (uint8_t)(((ch[9] << 2) | (ch[10] >> 9)) & 0xFF);
        buf[15] = (uint8_t)((ch[10] >> 1) & 0xFF);
        buf[16] = (uint8_t)(((ch[10] << 7) | (ch[11] >> 4)) & 0xFF);
        buf[17] = (uint8_t)(((ch[11] << 4) | (ch[12] >> 7)) & 0xFF);
        buf[18] = (uint8_t)(((ch[12] << 1) | (ch[13] >> 10)) & 0xFF);
        buf[19] = (uint8_t)((ch[13] >> 2) & 0xFF);
        buf[20] = (uint8_t)(((ch[13] << 6) | (ch[14] >> 5)) & 0xFF);
        buf[21] = (uint8_t)(((ch[14] << 3) | (ch[15] >> 8)) & 0xFF);
        buf[22] = (uint8_t)(ch[15] & 0xFF);

        // Flag
        buf[23] = 0x00;

        // 和校验 (Sum 0-23) & 0xFF
        unsigned int sum = 0;
        for(int i=0; i<24; i++) 
        {
            sum += buf[i];
        }
        buf[24] = (uint8_t)(sum & 0xFF);
    }

public:
    AirRobotMove()
    {
        ch_.resize(16);
        for(int i = 0; i < 16; i++)
        {
            ch_[i] = STOP.at(i);
        }
    }
    // {
    //     RCLCPP_INFO(this->get_logger(), "%s 节点开始运行", node_name);
    // }

    //port:串口文件路径/dev/...
    bool init(const std::string& port) override
    {
        serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if(serial_fd_ == -1)//打开失败
        {
            std::cerr << "串口打开失败" << std::endl;
            return false;
        }
        
// struct termios2 options;
//         // 获取当前配置 (使用 ioctl 替代 tcgetattr)
//         if (ioctl(serial_fd_, TCGETS2, &options) < 0) {
//             std::cerr << "[AirRobot] 无法获取串口配置 (TCGETS2)" << std::endl;
//             close(serial_fd_);
//             return false;
//         }

//        // 1. 设置非标准波特率: 100,000 bps
//         options.c_cflag &= ~CBAUD;   // 清除标准波特率掩码
//         options.c_cflag |= BOTHER;   // 启用自定义波特率标志
//         options.c_ispeed = 100000;   // 输入波特率 [修改处]
//         options.c_ospeed = 100000;   // 输出波特率 [修改处]

//         // 2. 设置数据位 8位 (CS8)
//         options.c_cflag &= ~CSIZE;
//         options.c_cflag |= CS8;

//         // 3. 设置校验位: 偶校验 (Even Parity) -> PARENB | !PARODD
//         options.c_cflag |= PARENB;  // 开启校验
//         options.c_cflag &= ~PARODD; // 关闭奇校验(即使用偶校验)
//         options.c_iflag |= (INPCK | ISTRIP); // 开启输入校验检查

//         // 4. 设置停止位: 2位 (CSTOPB)
//         options.c_cflag |= CSTOPB;

//         // 5. 其他标志
//         options.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器状态线，开启接收
//         options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 原始模式
//         options.c_oflag &= ~OPOST; // 原始输出
//         options.c_iflag &= ~(IXON | IXOFF | IXANY); // 关闭流控

//         // 应用设置 (使用 ioctl 替代 tcsetattr)
//         if (ioctl(serial_fd_, TCSETS2, &options) < 0) {
//             std::cerr << "[AirRobot] 无法应用串口配置 (TCSETS2)" << std::endl;
//             close(serial_fd_);
//             return false;
//         }


        return true;
    }

    //由于车身结构，前进和后退相反（气吸附1）
    const std::vector<uint16_t> BACKWARD   = {1500,1730,1500,1500,1,1,1,0, 0,0,1050,1950,0,0,0,0};
    const std::vector<uint16_t> FORWARD  = {1500,1270,1500,1500,1,1,1,0, 0,0,1050,1950,0,0,0,0};
    const std::vector<uint16_t> TURN_LEFT = {1200,1500,1500,1500,1,1500,1,1500, 1500,1500,1050,1950,0,0,0,0};
    const std::vector<uint16_t> TURN_RIGHT= {1900,1500,1500,1500,1,1500,1,1500, 1500,1500,1050,1950,0,0,0,0};
    const std::vector<uint16_t> STOP      = {1500,1500,1500,1500,1,1,1,0, 0,0,1050,1900,0,0,0,0};

    //   - 前进/后退：改 ch_[1]
    //   - 左转/右转：改 ch_[0]
    //   要求：输入 0.01 时，对应 Forward(1270), Backward(1730), Left(1200), Right(1900)
    void setVelocity(double linear_x, double vy, double angle_z) override
    {
        (void)vy;
        
        // --- 前后移动 channel 2 (数组索引1) ---
        // 目标：0.01输入 -> 变化量 230
        // 系数 = 230 / 0.01 = 23000
        const double LINEAR_SCALE = 23000.0;
        
        // 死区改为 0.001 以允许 0.01 的输入通过
        if(linear_x > 0.001) // 前进 -> 1270 (减小)
        {
            int val = 1500 - (int)(linear_x * LINEAR_SCALE);
            // 限制最小值，防止过度
            if(val <= 1050) val = 1050; 
            ch_[1] = (uint16_t)val;
        }
        else if(linear_x < -0.001) // 后退 -> 1730 (增加)
        {
            int val = 1500 + (int)(std::abs(linear_x) * LINEAR_SCALE);
            // 限制最大值
            if(val >= 1950) val = 1950;
            ch_[1] = (uint16_t)val;
        }
        else
        {
            ch_[1] = 1500;
        }

        // --- 转向 channel 1 (数组索引0) ---
        // 左转(>0): 0.01 -> 1200 (变化量 300) -> 系数 30000
        // 右转(<0): 0.01 -> 1900 (变化量 400) -> 系数 40000 (不对称)
        
        if(angle_z > 0.001) // 左转 -> 1200 (减小)
        {
            const double LEFT_SCALE = 30000.0;
            int val = 1500 - (int)(angle_z * LEFT_SCALE);
            
            if(val <= 1050) val = 1050;
            ch_[0] = (uint16_t)val;
        }
        else if(angle_z < -0.001) // 右转 -> 1900 (增加)
        {
            const double RIGHT_SCALE = 40000.0;
            int val = 1500 + (int)(std::abs(angle_z) * RIGHT_SCALE);
            
            if(val >= 1950) val = 1950;
            ch_[0] = (uint16_t)val;
        }
        else
        {
            ch_[0] = 1500;
        }
    }
    void send_loop() override
    {
        if(serial_fd_ != -1)
        {
            uint8_t buffer[25];
            pack_protocol_data(ch_, buffer);
            write(serial_fd_, buffer, 25);
        }
    }

    void moveshutdown() override
    {
        if(serial_fd_ != -1)
        {
            close(serial_fd_);
        }
    }
    // ~AirRobotMove()
    // {
    //     std::cout << "气吸附机器人停止工作!!!" << std::endl;
    // }
};
// ==========================================
// 5. 涵道气吸附机器人底盘控制实现 (UART/SBUS)
//    特点：中心位1000，200-1800
// ==========================================
class DuctRobotMove : public RobotMove
{
private:
    int serial_fd_ = -1;
    std::vector<uint16_t> ch_;

    // 核心打包函数：16通道 -> 25字节 (SBUS协议)
    // 核心打包函数：16通道 -> 25字节 (SBUS协议)
    void pack_protocol_data(std::vector<uint16_t> ch, uint8_t* buf) 
    {
        if (ch.size() != 16) 
        {
            std::cerr << "需要 16 个通道数据" << std::endl;
            return;
        }

        // 限制范围，确保每个通道数据不会超过 11bit 的极限 (2047)
        // 强制按位与 0x07FF 是为了防止移位时高位数据污染相邻通道
        for (int i = 0; i < 16; ++i) 
        {
            if(ch[i] > 1800) ch[i] = 1800; // 业务层面的安全限制
            ch[i] &= 0x07FF;               // 协议层面的安全截断 (11 bit)
        }

        buf[0] = 0x0F; // 帧头

        // 11bit 紧凑排列 (Little Endian logic - 匹配你的 STM32 Sbus_Data_Count 解析逻辑)
        // 通道 0-7
        buf[1]  = (uint8_t)(ch[0] & 0xFF);
        buf[2]  = (uint8_t)((ch[0] >> 8) | (ch[1] << 3));
        buf[3]  = (uint8_t)((ch[1] >> 5) | (ch[2] << 6));
        buf[4]  = (uint8_t)((ch[2] >> 2));
        buf[5]  = (uint8_t)((ch[2] >> 10) | (ch[3] << 1));
        buf[6]  = (uint8_t)((ch[3] >> 7) | (ch[4] << 4));
        buf[7]  = (uint8_t)((ch[4] >> 4) | (ch[5] << 7));
        buf[8]  = (uint8_t)((ch[5] >> 1));
        buf[9]  = (uint8_t)((ch[5] >> 9) | (ch[6] << 2));
        buf[10] = (uint8_t)((ch[6] >> 6) | (ch[7] << 5));
        buf[11] = (uint8_t)((ch[7] >> 3));

        // 通道 8-15
        buf[12] = (uint8_t)(ch[8] & 0xFF);
        buf[13] = (uint8_t)((ch[8] >> 8) | (ch[9] << 3));
        buf[14] = (uint8_t)((ch[9] >> 5) | (ch[10] << 6));
        buf[15] = (uint8_t)((ch[10] >> 2));
        buf[16] = (uint8_t)((ch[10] >> 10) | (ch[11] << 1));
        buf[17] = (uint8_t)((ch[11] >> 7) | (ch[12] << 4));
        buf[18] = (uint8_t)((ch[12] >> 4) | (ch[13] << 7));
        buf[19] = (uint8_t)((ch[13] >> 1));
        buf[20] = (uint8_t)((ch[13] >> 9) | (ch[14] << 2));
        buf[21] = (uint8_t)((ch[14] >> 6) | (ch[15] << 5));
        buf[22] = (uint8_t)((ch[15] >> 3));

        buf[23] = 0x00; // Flag 位，正常运行填 0 即可
        
        // 【重要修正】：SBUS没有校验和，结尾必须固定为 0x00
        buf[24] = 0x00; 
    }

public:
    DuctRobotMove()
    {
        ch_.resize(16, 1000);// 默认全给 1000 (安全中位)
        // 初始化状态
        // --- 匹配 STM32 下位机的初始化要求 ---
        ch_[0] = 1000; // CH1: 转向 (中位停止)
        ch_[1] = 1000; // CH2: 平移 (中位停止)未使用
        ch_[2] = 1000; // CH3: 前后 (中位停止)
        
        ch_[3] = 1000;  // CH4: 扫查器通信 (低电平关闭)
        ch_[4] = 1000;  // CH5: 扫查器电源 (低电平关闭)
        
        // CH6 必须 > 1300 才能使能底层电机的 PWM 输出
        ch_[5] = 1000; // CH6: 底盘总使能 (高电平开启)
        
        ch_[6] = 1600;  // CH7: MD2202 M1 (归中)
        ch_[7] = 1600;  // CH8: MD2202 M2 (归中)

        ch_[8]  = 60;    // CH9
        ch_[9]  = 30; // CH10
        ch_[10] = 3; // CH11
    }

    bool init(const std::string& port) override
    {
        std::cout << "[DuctRobot] 正在打开涵道串口: " << port << std::endl;
        serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if(serial_fd_ == -1)
        {
            std::cerr << "涵道串口打开失败" << std::endl;
            return false;
        }
        
    //     struct termios2 options;
    //     // 获取当前配置 (使用 ioctl 替代 tcgetattr)
    //     if (ioctl(serial_fd_, TCGETS2, &options) < 0) {
    //         std::cerr << "[DuctRobot] 无法获取串口配置 (TCGETS2)" << std::endl;
    //         close(serial_fd_);
    //         return false;
    //     }

    //    // 1. 设置非标准波特率: 100,000 bps
    //     options.c_cflag &= ~CBAUD;   // 清除标准波特率掩码
    //     options.c_cflag |= BOTHER;   // 启用自定义波特率标志
    //     options.c_ispeed = 100000;   // 输入波特率 [修改处]
    //     options.c_ospeed = 100000;   // 输出波特率 [修改处]

    //     // 2. 设置数据位 8位 (CS8)
    //     options.c_cflag &= ~CSIZE;
    //     options.c_cflag |= CS8;

    //     // 3. 设置校验位: 偶校验 (Even Parity) -> PARENB | !PARODD
    //     options.c_cflag |= PARENB;  // 开启校验
    //     options.c_cflag &= ~PARODD; // 关闭奇校验(即使用偶校验)
    //     // options.c_iflag |= (INPCK | ISTRIP); // 开启输入校验检查

    //     // 4. 设置停止位: 2位 (CSTOPB)
    //     options.c_cflag |= CSTOPB;

    //     // 5. 其他标志
    //     options.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器状态线，开启接收
    //     options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 原始模式
    //     options.c_oflag &= ~OPOST; // 原始输出
    //     options.c_iflag &= ~(IXON | IXOFF | IXANY); // 关闭流控

    //     // 应用设置 (使用 ioctl 替代 tcsetattr)
    //     if (ioctl(serial_fd_, TCSETS2, &options) < 0) {
    //         std::cerr << "[DuctRobot] 无法应用串口配置 (TCSETS2)" << std::endl;
    //         close(serial_fd_);
    //         return false;
    //     }
    //     std::cout << "[DuctRobot] 第一阶段：使能步进电机..." << std::endl;
    //     ch_[5] = 1600; // 先只开步进电机
    //     for(int i=0; i<5; i++) {
    //         send_loop();
    //         usleep(1500); // 延时 200ms
    //     }

    //     std::cout << "[DuctRobot] 第二阶段：关闭扫查器电源继电器..." << std::endl;
    //     ch_[4] = 500; // 稳定后再开继电器
    //     for(int i=0; i<5; i++) {
    //         send_loop();
    //         usleep(1500); // 延时 200ms
    //     }

    //     std::cout << "[DuctRobot] 第三阶段：打开扫查器电源继电器..." << std::endl;
    //     ch_[4] = 1600; // 稳定后再开继电器
    //     for(int i=0; i<5; i++) {
    //         send_loop();
    //         usleep(1500); // 延时 200ms
    //     }
        
        std::cout << "[DuctRobot] 硬件初始化与上电完成！" << std::endl;
        return true;
    }

    /*
     * 速度控制逻辑 (严格对应 MC6RE STM32逻辑):
     * - 中心停止位: 1000
     * - CH1(Index 0) 转向: <700 左转, >1300 右转
     * - CH2(Index 1) 平移: <700 右平移, >1300 左平移
     * - CH3(Index 2) 前后: <700 后退, >1300 前进
     */
    void setVelocity(double linear_x, double vy, double angle_z) override
    {
        // 1. 禁用平移 (锁定 CH2 为中位)
        (void)vy; 
        const double SCALE = 1000.0;

        // 2. 前后移动 (CH3)
        if (linear_x >= 0.01) // 前进 (输入 >= 0.01)
        {
            ch_[2] = 1600;
        }
        else if (linear_x <= -0.01) // 后退 (输入 <= -0.01)
        {

            ch_[2] = 500;
        }
        else
        {
            ch_[2] = 1000; // 停止 (输入 < 0.01)
        }

        // ==========================
        // 2. 转向移动 (CH1) - 同样逻辑
        // Python脚本: 左(Low < 1350), 右(High > 1650)
        // ==========================
        if (angle_z >= 0.01) // 左转
        {
            ch_[0] = 500;
        }
        else if (angle_z <= -0.01) // 右转
        {
             ch_[0] = 1600;
        }
        else
        {
            ch_[0] = 1000;
        }
        ch_[5] = 1600;
    }

    void setScannerConfig(double speed, double distance, double precision) override
    {
        // 1. 扫查速度映射: 上位机 0.5~10 映射到 scan_speed 60~120
        uint16_t scan_speed = 60;
        if (speed >= 10.0) 
        {
            scan_speed = 120;
        }
        else if (speed >= 0.5) 
        {
            scan_speed = 60 + (uint16_t)((speed - 0.5) * 60.0 / 9.5);
        }
        // 2. 扫查距离映射: 上位机 0~50
        uint16_t scan_distance = (uint16_t)distance;
        if (scan_distance > 50) scan_distance = 50;

        // 3. 扫查精度映射: 0.01~1 映射到分段数
        uint16_t scan_segments = 3; 
        if (precision > 0.001 && precision <= 1.0) scan_segments = (uint16_t)(1.0 / precision);

        // 赋值给 SBUS 通道
        ch_[8]  = scan_speed;    // CH9
        ch_[9]  = scan_distance; // CH10
        ch_[10] = scan_segments; // CH11
    }

    void scannercontrol() override
    {
        ch_[6] = 1000;  // CH7: MD2202 M1 (归中)
        ch_[3] = 1600;
        for(int i=0; i<5; i++) {
            send_loop();
            usleep(20000);
        }
        // 2. 监听串口，等待反馈帧 0xAA 0x20 0x00 0x55
        std::vector<uint8_t> rx_buffer;
        bool target_received = false;
        
        // // 设置超时机制，防止死等 (例如最多等待 2秒 = 100次 * 20ms)
        // for(int try_cnt = 0; try_cnt < 100; try_cnt++) {
        //     uint8_t temp_buf[64];
        //     // O_NDELAY 模式下 read 是非阻塞的
        //     int bytes_read = read(serial_fd_, temp_buf, sizeof(temp_buf));
        //     if (bytes_read > 0) {
        //         // 将新读到的数据拼接到缓存末尾（防止指令被截断成两半）
        //         rx_buffer.insert(rx_buffer.end(), temp_buf, temp_buf + bytes_read);
                
        //         // 滑动窗口检查是否存在目标序列
        //         if (rx_buffer.size() >= 4) {
        //             for (size_t i = 0; i <= rx_buffer.size() - 4; i++) {
        //                 if (rx_buffer[i] == 0xAA && rx_buffer[i+1] == 0x20 && 
        //                     rx_buffer[i+2] == 0x00 && rx_buffer[i+3] == 0x55) {
        //                     target_received = true;
        //                     break;
        //                 }
        //             }
        //         }
        //     }
        //     if(target_received)
        //     {
        //         break;
        //     }
        //     usleep(20000); // 没收到则休眠 20ms 后重试
        // }

        // // 3. 如果收到指定数据，车体前进一段时间
        // if (target_received) {
        //     std::cout << "[ScannerControl] 收到反馈指令 0xAA 0x20 0x00 0x55,车体开始前进!" << std::endl;
            
        //     ch_[3] = 500; 
            
        //     // 持续发送前进指令一段时间 (例如 1秒 = 50次 * 20ms)
        //     for(int i = 0; i < 50; i++) {
        //         send_loop();
        //         usleep(20000);
        //     }
            
        //     // // 4. 前进结束，恢复底盘停止状态 (中位 1000)
        //     // // 如果是 AirRobotMove 中使用，这里应改为 ch_[1] = 1500;
        //     // ch_[1] = 1000; 
        //     // for(int i = 0; i < 5; i++) {
        //     //     send_loop();
        //     //     usleep(15000);
        //     // }
        //     // std::cout << "[ScannerControl] 前进动作完成，底盘已停止。" << std::endl;
        // } else {
        //     std::cout << "[ScannerControl] 等待反馈指令超时。" << std::endl;
        // }
        ch_[3] = 1000;
    }

    void send_loop() override
    {
        if(serial_fd_ != -1)
        {
            uint8_t buffer[25];
            pack_protocol_data(ch_, buffer);
           // 记录返回值
            ssize_t written = write(serial_fd_, buffer, 25);
            
            // 打印调试信息 (为了防止刷屏，可以每50次打印一次)
            static int debug_cnt = 0;
            // 每 50 次循环打印一次 (防止刷屏太快无法看清)，如果需要看每一帧，请注释掉 if
            if(debug_cnt++ % 50 == 0) 
            {
                std::cout << "[SBUS Hex]: ";
                for(int i = 0; i < 25; i++) 
                {
                    // %02X 表示输出2位十六进制，不足补0，大写
                    printf("%02X ", buffer[i]); 
                }
                std::cout << std::endl;
            }
        }
    }
    void moveshutdown() override
    {
        if(serial_fd_ != -1)
        {
            // 1. 归中运动轴
            ch_[0] = 1000;
            ch_[1] = 1000;
            ch_[2] = 1000;
            // 2. 关闭扫查器外设
            ch_[3] = 500; 
            ch_[4] = 500;
            // 3. 失能底盘步进电机 (CH6 < 700)
            ch_[5] = 500;
            // 连发几帧确保收到
            for(int i=0; i<5; i++) {
                send_loop();
                usleep(20000); // 20ms
            }

            close(serial_fd_);
            serial_fd_ = -1;
            std::cout << "[DuctRobot] 涵道机器人已停止、步进电机脱机" << std::endl;
        }
    }
};
// ==========================================
// 3. 磁吸附机器人底盘控制实现 (SBUS 协议)
//    100k baud(approx), 8E2
// ==========================================
class MagRobotMove : public RobotMove
{
private:
    int serial_fd_ = -1;
    std::vector<uint16_t> ch_; // 存储16个通道的值

    // SBUS 核心打包函数
    void pack_protocol_data(const std::vector<uint16_t>& ch, uint8_t* buf) 
    {
        if (ch.size() != 16) return;

        // 限制范围 [0, 2047]
        std::vector<uint16_t> clean_ch = ch;
        for(auto& val : clean_ch) {
            if(val > 2047) val = 2047;
        }

        buf[0] = 0x0F; // Header

        // 11 bits per channel packing
        buf[1]  = (uint8_t)((clean_ch[0] & 0x07FF));
        buf[2]  = (uint8_t)((clean_ch[0] & 0x07FF) >> 8 | (clean_ch[1] & 0x07FF) << 3);
        buf[3]  = (uint8_t)((clean_ch[1] & 0x07FF) >> 5 | (clean_ch[2] & 0x07FF) << 6);
        buf[4]  = (uint8_t)((clean_ch[2] & 0x07FF) >> 2);
        buf[5]  = (uint8_t)((clean_ch[2] & 0x07FF) >> 10 | (clean_ch[3] & 0x07FF) << 1);
        buf[6]  = (uint8_t)((clean_ch[3] & 0x07FF) >> 7 | (clean_ch[4] & 0x07FF) << 4);
        buf[7]  = (uint8_t)((clean_ch[4] & 0x07FF) >> 4 | (clean_ch[5] & 0x07FF) << 7);
        buf[8]  = (uint8_t)((clean_ch[5] & 0x07FF) >> 1);
        buf[9]  = (uint8_t)((clean_ch[5] & 0x07FF) >> 9 | (clean_ch[6] & 0x07FF) << 2);
        buf[10] = (uint8_t)((clean_ch[6] & 0x07FF) >> 6 | (clean_ch[7] & 0x07FF) << 5);
        buf[11] = (uint8_t)((clean_ch[7] & 0x07FF) >> 3);
        buf[12] = (uint8_t)((clean_ch[8] & 0x07FF));
        buf[13] = (uint8_t)((clean_ch[8] & 0x07FF) >> 8 | (clean_ch[9] & 0x07FF) << 3);
        buf[14] = (uint8_t)((clean_ch[9] & 0x07FF) >> 5 | (clean_ch[10] & 0x07FF) << 6);
        buf[15] = (uint8_t)((clean_ch[10] & 0x07FF) >> 2);
        buf[16] = (uint8_t)((clean_ch[10] & 0x07FF) >> 10 | (clean_ch[11] & 0x07FF) << 1);
        buf[17] = (uint8_t)((clean_ch[11] & 0x07FF) >> 7 | (clean_ch[12] & 0x07FF) << 4);
        buf[18] = (uint8_t)((clean_ch[12] & 0x07FF) >> 4 | (clean_ch[13] & 0x07FF) << 7);
        buf[19] = (uint8_t)((clean_ch[13] & 0x07FF) >> 1);
        buf[20] = (uint8_t)((clean_ch[13] & 0x07FF) >> 9 | (clean_ch[14] & 0x07FF) << 2);
        buf[21] = (uint8_t)((clean_ch[14] & 0x07FF) >> 6 | (clean_ch[15] & 0x07FF) << 5);
        buf[22] = (uint8_t)((clean_ch[15] & 0x07FF) >> 3);

        buf[23] = 0x00; // Flag
        buf[24] = 0x00; // End byte (SBUS standard is 0x00)
    }

    // 常用数值定义
    const uint16_t VAL_LOW  = 362;
    const uint16_t VAL_MID  = 1002;
    const uint16_t VAL_HIGH = 1642;

public:
    MagRobotMove()
    {
        // 初始化通道: 全部置中位 1500
        ch_.resize(16, VAL_MID);
    }

    bool init(const std::string& port) override
    {
        std::cout << "[MagRobot] 正在打开串口: " << port << " (115200, 8E1)..." << std::endl;
        serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if(serial_fd_ == -1)
        {
            std::cerr << "[MagRobot] 串口打开失败!" << std::endl;
            return false;
        }
        
    //     struct termios2 options;
    //     // 获取当前配置 (使用 ioctl 替代 tcgetattr)
    //     if (ioctl(serial_fd_, TCGETS2, &options) < 0) {
    //         std::cerr << "[MagRobot] 无法获取串口配置 (TCGETS2)" << std::endl;
    //         close(serial_fd_);
    //         return false;
    //     }

    //    // 1. 设置非标准波特率: 100,000 bps
    //     options.c_cflag &= ~CBAUD;   // 清除标准波特率掩码
    //     options.c_cflag |= B115200;   // 启用自定义波特率标志 [重点修改]
    //     options.c_ispeed = 115200;   // 设置输入波特率为 100000 [修改处]
    //     options.c_ospeed = 115200;   // 设置输出波特率为 100000 [修改处]

    //     // 2. 设置数据位 8位 (CS8)
    //     options.c_cflag &= ~CSIZE;
    //     options.c_cflag |= CS8;

    //     // 3. 关闭校验位 (None Parity) - 匹配下位机 8N1
    //     options.c_cflag &= ~PARENB;  
    //     options.c_iflag &= ~(INPCK | ISTRIP); 

    //     // 4. 设置停止位: 1位 - 匹配下位机 8N1
    //     options.c_cflag &= ~CSTOPB;

    //     // 5. 其他标志
    //     options.c_cflag |= (CLOCAL | CREAD); // 忽略调制解调器状态线，开启接收
    //     options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 原始模式
    //     options.c_oflag &= ~OPOST; // 原始输出
    //     options.c_iflag &= ~(IXON | IXOFF | IXANY); // 关闭流控

    //     // 应用设置 (使用 ioctl 替代 tcsetattr)
    //     if (ioctl(serial_fd_, TCSETS2, &options) < 0) {
    //         std::cerr << "[MagRobot] 无法应用串口配置 (TCSETS2)" << std::endl;
    //         close(serial_fd_);
    //         return false;
    //     }

        // 初始化特定通道状态 (参考 Python 脚本)
        // CH7 (Index 6): Enable -> High (解锁)
        ch_[6] = VAL_HIGH;
        // CH8 (Index 7): Speed Mode -> Low (低速)
        ch_[7] = VAL_LOW; 
        std::cout << "[MagRobot] 发送解锁信号 (CH7 -> High)..." << std::endl;

        // 连续发送 20 帧，确保接收机接收到跳变信号
        for(int i=0; i<20; i++) {
            send_loop();
            usleep(14000);
        }
        std::cout << "[MagRobot] 初始化完成: 电机已解锁 (CH7=High), 速度设为低速 (CH8=Low)" << std::endl;
        return true;
    }

    // 设置速度
    // ---------------------------------------------------------
    // 针对 ±150 硬件死区的特殊映射
    // 逻辑：输入 0.01 时，PWM 立即跳跃到 1656 (刚好起步)
    // ---------------------------------------------------------
    void setVelocity(double linear_x, double vy, double angle_z) override
    {
        (void)vy; // 磁吸附不支持横移

        // 硬件死区：1500 ± 150 是电机不动的区间
        const uint16_t HW_DEAD_ZONE = 150; 

        // ==========================
        // 1. 前后移动 (CH2)
        // ==========================
        if (linear_x >= 0.01) // 前进 (输入 >= 0.01)
        {
            ch_[1] = 1642;
        }
        else if (linear_x <= -0.01) // 后退 (输入 <= -0.01)
        {

            ch_[1] = 362;
        }
        else
        {
            ch_[1] = 1002; // 停止 (输入 < 0.01)
        }

        // ==========================
        // 2. 转向移动 (CH1) - 同样逻辑
        // Python脚本: 左(Low < 1350), 右(High > 1650)
        // ==========================
        if (angle_z >= 0.01) // 左转
        {
            ch_[0] = 362;
        }
        else if (angle_z <= -0.01) // 右转
        {
             ch_[0] = 1642;
        }
        else
        {
            ch_[0] = 1002;
        }
    }

    void send_loop() override
    {
        if(serial_fd_ != -1)
        {
            uint8_t buffer[25];
            pack_protocol_data(ch_, buffer);
           // 记录返回值
            ssize_t written = write(serial_fd_, buffer, 25);
            
            // 打印调试信息 (为了防止刷屏，可以每50次打印一次)
            static int debug_cnt = 0;
            // 每 50 次循环打印一次 (防止刷屏太快无法看清)，如果需要看每一帧，请注释掉 if
            if(debug_cnt++ % 50 == 0) 
            {
                std::cout << "[SBUS Hex]: ";
                for(int i = 0; i < 25; i++) 
                {
                    // %02X 表示输出2位十六进制，不足补0，大写
                    printf("%02X ", buffer[i]); 
                }
                std::cout << std::endl;
            }
        }
    }

    void moveshutdown() override
    {
        if(serial_fd_ != -1)
        {
            // 停机前动作: 归中并上锁
            std::fill(ch_.begin(), ch_.end(), VAL_MID);
            ch_[6] = VAL_LOW; // Disable
            
            // 发送几次停止帧
            for(int i=0; i<5; i++) {
                send_loop();
                usleep(20000);
            }
            
            close(serial_fd_);
            serial_fd_ = -1;
            std::cout << "[MagRobot] 已停止并关闭串口。" << std::endl;
        }
    }
};
// ==========================================
// 4. 仙工 AGV 实现 (多端口API + 抱闸解锁)TCP/IP
// Winsock-->BSD Socket   SOCKET-->int
// ==========================================
class AGVRobotMove : public RobotMove
{
private:
    int sock_19204_, sock_19205_, sock_19206_, sock_19210_;
    uint16_t seq_num_ = 0;
    bool connectPort(const std::string& ip, int port, int& sock_fd)
    {
        //1、创建套接字
        sock_fd = socket(AF_INET, SOCK_STREAM, 0);//创建 socket,IPv4 / TCP
        if (sock_fd < 0) 
        {
            return false;
        }

        //2、设置地址结构
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr)); // 清空结构体
        addr.sin_family = AF_INET; // IPv4
        addr.sin_port = htons(port);    // 端口 (网络字节序)
        //将点分十进制格式的地址字符串转换为网络字节序整型数
        if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0) // 设置服务器IP,成功返回1
        {
            return false;
        }

        //3、连接
        struct timeval timeout;      
        timeout.tv_sec = 1; timeout.tv_usec = 0;// 设置超时时间 1 秒
        setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout,sizeof(timeout));// 设置接收超时
        setsockopt(sock_fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout,sizeof(timeout));// 设置发送超时

        if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) //连接失败返回-1
        {
            close(sock_fd);//关闭socket
            sock_fd = -1;
            return false;
        }
        return true;
    }

    void unlockBrake() 
    {
        // ID: 6001, Content: {"id":0,"status":true}
        // std::string json = "{\"id\":0,\"status\":true}";
        RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "进入！");
        cJSON *root = cJSON_CreateObject();           // 创建JSON对象
        cJSON_AddNumberToObject(root, "id", 0);        // 添加刹车ID字段
        cJSON_AddBoolToObject(root, "status", true);   // 设置状态为解锁
        char* json_str = cJSON_PrintUnformatted(root); // 序列化为字符串
        if(json_str)
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "协议发送！");
            std::vector<uint8_t> frame = packSeerFrame(6001, json_str); // 封装为seer通信帧
            if (sock_19210_ >= 0) 
            {
                send(sock_19210_, frame.data(), frame.size(), 0); // 发送解锁命令
            }
            free(json_str);                            // 释放JSON字符串内存
        }
        cJSON_Delete(root);                            // 释放JSON对象内存
    }
    //将纯json字符串包装成seer的通信帧
    std::vector<uint8_t> packSeerFrame(uint16_t type, const std::string& json) 
    {
        uint32_t len = json.length();
        std::vector<uint8_t> frame(16 + len);

        frame[0] = 0x5A; // Head
        frame[1] = 0x01; // Version
        
        seq_num_++; // Seq Num (Big Endian)
        frame[2] = (seq_num_ >> 8) & 0xFF; 
        frame[3] = seq_num_ & 0xFF;

        frame[4] = (len >> 24) & 0xFF;     // Length (Big Endian)
        frame[5] = (len >> 16) & 0xFF;
        frame[6] = (len >> 8) & 0xFF;
        frame[7] = len & 0xFF;

        frame[8] = (type >> 8) & 0xFF;     // Type (Big Endian)
        frame[9] = type & 0xFF;

        memset(&frame[10], 0, 6);          // Reserved
        memcpy(&frame[16], json.c_str(), len); // Body
        return frame;
    }
    //读取响应
    void readresponse(int sock_fd, const std::string& cmd_name) 
    {
        if (sock_fd < 0) return;
        char buffer[4096]; 
        memset(buffer, 0, sizeof(buffer));
        int len = recv(sock_fd, buffer, sizeof(buffer)-1, 0);
        if (len > 16) 
        {
            std::cout << "[AGV] " << cmd_name << " Reply: " << (buffer+16) << std::endl;
           
        }
    }
    // 查询当前坐标 (ID: 1004)
    bool getcurrentpose(double& x, double& y, double& theta)
    {
        if (sock_19204_ < 0) return false;
        //
        std::vector<uint8_t> frame = packSeerFrame(1004, "{}");
        send(sock_19204_, frame.data(), frame.size(), 0);

        char buffer[4096]; 
        memset(buffer, 0, sizeof(buffer));
        int len = recv(sock_19204_, buffer, sizeof(buffer)-1, 0);
        if(len > 16)
        {
            char *resp_data = buffer + 16;
            cJSON *json_data = cJSON_Parse(resp_data);//解析数据
            if(!json_data)
            {
                std::cerr << "[AGV] 位置响应数据解析失败！！！" << std::endl;
                return false;
            }
            cJSON *Item_x = cJSON_GetObjectItem(json_data, "x");
            cJSON *Item_y = cJSON_GetObjectItem(json_data, "y");
            cJSON *Item_angle = cJSON_GetObjectItem(json_data, "angle");

            bool success = false;
            if(Item_x && Item_y && Item_angle)
            {
                if(Item_x->type == cJSON_Number && Item_y->type == cJSON_Number && Item_angle->type == cJSON_Number )
                {
                    x = Item_x->valuedouble;
                    y = Item_y->valuedouble;
                    theta = Item_angle->valuedouble;
                    success = true;
                }
            }
            else
            {
                 std::cerr << "[AGV] JSON x/y/angle 数据错误！！！" << std::endl;
            }

            cJSON_Delete(json_data);
            return success;
        }
        return false;
    }
    //开机重定位
    void robot_control_reloc_req(void)
    {
        double x = 0.0, y = 0.0, theta = 0.0;
        if(!getcurrentpose(x, y, theta))
        {
            std::cerr << "[AGV] 重定位失败：获取当前位置失败!!!" << std::endl;
            return;
        }
        std::cout << "[AGV] 当前位置: x=" << x << ", y=" << y << ", theta=" << theta << std::endl;
        cJSON *root = cJSON_CreateObject();
        cJSON_AddNumberToObject(root, "x", x);
        cJSON_AddNumberToObject(root, "y", y);
        cJSON_AddNumberToObject(root, "theta", theta);
        cJSON_AddNumberToObject(root, "length", 2); // 搜索范围长宽 2m
        char* json_str = cJSON_PrintUnformatted(root);
        if(json_str)
        {
            std::vector<uint8_t> frame = packSeerFrame(2002, json_str);
            if (sock_19205_ >= 0) 
            {
                send(sock_19205_, frame.data(), frame.size(), 0);
            }
            free(json_str);
        }
        cJSON_Delete(root);
        std::cout << "[AGV] 等待重定位(3s)..." << std::endl;
        sleep(3);

        std::vector<uint8_t> comfirm_frame = packSeerFrame(2003, "{}");
        send(sock_19205_, comfirm_frame.data(), comfirm_frame.size(), 0);

        readresponse(sock_19205_, "Reloc Confirm (2003)");
        std::cout << "[AGV] 重定位成功!" << std::endl;
    }
    //导航状态读取
    bool robot_navigation_status()
    {
        if(sock_19204_ < 0) return false;
        std::cout << "[AGV] 等待导航任务完成..." << std::endl;
        while(true)
        {
            cJSON *navi_root = cJSON_CreateObject();
            cJSON_AddBoolToObject(navi_root, "simple", true);
            char* str = cJSON_PrintUnformatted(navi_root);
            //发送导航状态查询请求
            if(str)
            {
                std::vector<uint8_t> navi_frame = packSeerFrame(1020, str);
                send(sock_19204_, navi_frame.data(), navi_frame.size(), 0);
                free(str);
            }
            cJSON_Delete(navi_root);

            // 接收响应
            char buf[4096];
            int len = recv(sock_19204_, buf, 4095, 0);
            //解析响应
            if(len > 16)
            {
                cJSON *resp_data = cJSON_Parse(buf + 16);
                if(!resp_data)
                {
                    std::cerr << "[AGV] 导航响应数据解析失败！！！" << std::endl;
                    cJSON_Delete(resp_data);
                    return false;
                }
                cJSON *task_status = cJSON_GetObjectItem(resp_data, "task_status");
                if(task_status)
                {
                    int status = task_status->valueint;
                    // 0:空闲/结束, 4:完成
                    if (status == 0 || status == 4)
                    {
                        cJSON_Delete(resp_data);
                        std::cout << "[AGV] 导航完成 (Status: " << status << ")" << std::endl;
                        return true;
                    }
                    // 5:失败 (可选: 遇到阻挡是否退出?)
                    if (status == 5) 
                    {
                        cJSON_Delete(resp_data);
                        std::cerr << "[AGV] 导航失败 (Status: 5)!" << std::endl;
                        return false;
                    }
                }
                cJSON_Delete(resp_data);
            }
            // 必须休眠，否则 CPU 100%
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

public:
    AGVRobotMove() 
    {
        sock_19204_ = -1; //查询
        sock_19205_ = -1; //控制
        sock_19206_ = -1; //导航
        sock_19210_ = -1; //DO 抱闸
    }
    // 初始化: 连接 4 个端口并解锁抱闸，可输入IP:port(如"192.168.1.10:19204")
    bool init(const std::string& seer_ip) override
    {
        RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "初始化连接");
        std::string ip = seer_ip;
        size_t colon = seer_ip.find(':');
        //截取ip
        if (colon != std::string::npos) 
        {
            ip = seer_ip.substr(0, colon);
        }
        std::cout << "[AGV] 初始化连接到 " << ip << "..." << std::endl;

        bool s10 = connectPort(ip, 19210, sock_19210_); // IO / 抱闸
        bool s04 = connectPort(ip, 19204, sock_19204_); // 查询
        bool s05 = connectPort(ip, 19205, sock_19205_); // 控制
        bool s06 = connectPort(ip, 19206, sock_19206_); // 导航

        // 只要控制端口能通，就认为初始化成功，但打印警告
        if (s10) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19210成功");
            std::cout << "[AGV]连接19210成功..." << std::endl;
        }
        else
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19210shibai");
            std::cerr << "[AGV]连接19210失败..." << std::endl;
        }
        if (s04) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19204成功");
            std::cout << "[AGV]连接19204成功..." << std::endl;
        } 
        else 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19204");
            std::cerr << "[AGV]连接19204失败..." << std::endl;
        }
        if (s05) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19205成功");
            std::cout << "[AGV]连接19205成功..." << std::endl;
        } 
        else 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19210");
            std::cerr << "[AGV]连接19205失败..." << std::endl;
            return false;
        }
        if (s06) 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19206成功");
            std::cout << "[AGV]连接19206成功..." << std::endl;
        } 
        else 
        {
            RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]连接19206");
            std::cerr << "[AGV]连接19206失败..." << std::endl;
        }
        unlockBrake();
        RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV] 初始化端口结束！");
        std::cout << "[AGV] 初始化端口结束！" << std::endl;

        // 执行自动重定位
        RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV] 开始自动重定位！");
        std::cout << "[AGV] 开始自动重定位..." << std::endl;
        robot_control_reloc_req();
        RCLCPP_ERROR(rclcpp::get_logger("AgvHardwareInterface"),
                     "[AGV]重定位！");
        return true;
    }

    bool getpose(double& x, double& y, double& theta) override
    {
        return getcurrentpose(x, y, theta);
    }
    
    void setVelocity(double vx, double vy, double wz) override
    {
        if (sock_19205_ < 0) return;
        cJSON *root = cJSON_CreateObject();
        cJSON_AddNumberToObject(root, "vx", vx);
        cJSON_AddNumberToObject(root, "vy", vy);
        cJSON_AddNumberToObject(root, "w", wz);
        cJSON_AddNumberToObject(root, "duration", 0); // 持续时间0表示一直保持直到下个指令
        char* json_str = cJSON_PrintUnformatted(root);
        
        if(json_str)
        {
            std::vector<uint8_t> frame = packSeerFrame(2010, json_str);
            send(sock_19205_, frame.data(), frame.size(), 0);
            free(json_str);
        }

        cJSON_Delete(root);
    }
    
    //平动 ID：3055
    bool movebydistance(double dist, double vx, double vy) override
    { 
        if(sock_19206_ < 0) return false; 
        std::cout << "[AGV] 直线运动距离: " << dist << "m" << std::endl;

        cJSON *root = cJSON_CreateObject();
        cJSON_AddNumberToObject(root, "dist", dist);//m
        cJSON_AddNumberToObject(root, "vx", vx);
        cJSON_AddNumberToObject(root, "vy", vy);
        char* json_str = cJSON_PrintUnformatted(root);
        if(json_str)
        {
            std::vector<uint8_t> frame = packSeerFrame(3055, json_str);
            send(sock_19206_, frame.data(), frame.size(), 0);
            free(json_str);
            // 简单读取响应
            readresponse(sock_19206_, "平动");
            return robot_navigation_status();
        }
        cJSON_Delete(root);
        return false;
    }
    //转动
    bool rotatebyangle(double angle, double vw) override
    { 
        if(sock_19206_ < 0) return false; 
        std::cout << "[AGV] 转动角度: " << angle << "rad" << std::endl;

        cJSON *root = cJSON_CreateObject();
        cJSON_AddNumberToObject(root, "angle", angle);//m
        cJSON_AddNumberToObject(root, "vw", vw);
        char* json_str = cJSON_PrintUnformatted(root);
        if(json_str)
        {
            std::vector<uint8_t> frame = packSeerFrame(3056, json_str);
            send(sock_19206_, frame.data(), frame.size(), 0);
            free(json_str);
            // 简单读取响应
            readresponse(sock_19206_, "转动");
            return robot_navigation_status();
        }
        cJSON_Delete(root);
        return false;
    }

    
    void send_loop() override{}

    void moveshutdown() override
    {
        if (sock_19204_ >= 0) close(sock_19204_);
        if (sock_19205_ >= 0) close(sock_19205_);
        if (sock_19206_ >= 0) close(sock_19206_);
        if (sock_19210_ >= 0) close(sock_19210_);

        std::cout << "AGV 已停止运行！！！" << std::endl;
    }
};
#endif