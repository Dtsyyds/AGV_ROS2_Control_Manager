#include "robot_move/impl/duct/duct_robot_move.hpp"

#include <asm/termbits.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>

DuctRobotMove::DuctRobotMove() : serial_fd_(-1), ch_(16, 1000)
{
    ch_[0] = 1000;
    ch_[1] = 1000;
    ch_[2] = 1000;
        
    ch_[3] = 1000;
    ch_[4] = 1000;
        
    ch_[5] = 1000;
        
    ch_[6] = 1600;
    ch_[7] = 1600;

    ch_[8]  = 60;
    ch_[9]  = 30;
    ch_[10] = 3;
}

bool DuctRobotMove::init(const std::string& port)
{
    std::cout << "[DuctRobot] 正在打开涵道串口: " << port << std::endl;
    serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if(serial_fd_ == -1)
    {
        std::cerr << "涵道串口打开失败" << std::endl;
        return false;
    }
        
    struct termios2 options;
    if (ioctl(serial_fd_, TCGETS2, &options) < 0) {
        std::cerr << "[DuctRobot] 无法获取串口配置 (TCGETS2)" << std::endl;
        close(serial_fd_);
        return false;
    }

    options.c_cflag &= ~CBAUD;
    options.c_cflag |= BOTHER;
    options.c_ispeed = 100000;
    options.c_ospeed = 100000;

    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    options.c_cflag |= PARENB;
    options.c_cflag &= ~PARODD;

    options.c_cflag |= CSTOPB;

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    if (ioctl(serial_fd_, TCSETS2, &options) < 0) {
        std::cerr << "[DuctRobot] 无法应用串口配置 (TCSETS2)" << std::endl;
        close(serial_fd_);
        return false;
    }
    std::cout << "[DuctRobot] 第一阶段：使能步进电机..." << std::endl;
    ch_[5] = 1600;
    for(int i=0; i<5; i++) {
        send_loop();
        usleep(1500);
    }

    std::cout << "[DuctRobot] 第二阶段：关闭扫查器电源继电器..." << std::endl;
    ch_[4] = 500;
    for(int i=0; i<5; i++) {
        send_loop();
        usleep(1500);
    }

    std::cout << "[DuctRobot] 第三阶段：打开扫查器电源继电器..." << std::endl;
    ch_[4] = 1600;
    for(int i=0; i<5; i++) {
        send_loop();
        usleep(1500);
    }
        
    std::cout << "[DuctRobot] 硬件初始化与上电完成！" << std::endl;
    return true;
}

void DuctRobotMove::setVelocity(double linear_x, double vy, double angle_z)
{
    (void)vy;

    if (linear_x >= 0.01)
    {
        ch_[2] = 1600;
    }
    else if (linear_x <= -0.01)
    {
        ch_[2] = 500;
    }
    else
    {
        ch_[2] = 1000;
    }

    if (angle_z >= 0.01)
    {
        ch_[0] = 500;
    }
    else if (angle_z <= -0.01)
    {
        ch_[0] = 1600;
    }
    else
    {
        ch_[0] = 1000;
    }
    ch_[5] = 1600;
}

void DuctRobotMove::setScannerConfig(double speed, double distance, double precision)
{
    uint16_t scan_speed = 60;
    if (speed >= 10.0) 
    {
        scan_speed = 120;
    }
    else if (speed >= 0.5) 
    {
        scan_speed = 60 + (uint16_t)((speed - 0.5) * 60.0 / 9.5);
    }
    uint16_t scan_distance = (uint16_t)distance;
    if (scan_distance > 50) scan_distance = 50;

    uint16_t scan_segments = 3; 
    if (precision > 0.001 && precision <= 1.0) scan_segments = (uint16_t)(1.0 / precision);

    ch_[8]  = scan_speed;
    ch_[9]  = scan_distance;
    ch_[10] = scan_segments;
}

void DuctRobotMove::scannercontrol()
{
    ch_[6] = 1000;
    ch_[3] = 1600;
    for(int i=0; i<5; i++) {
        send_loop();
        usleep(20000);
    }
    ch_[3] = 1000;
}

void DuctRobotMove::send_loop()
{
    if(serial_fd_ != -1)
    {
        uint8_t buffer[25];
        pack_protocol_data(ch_, buffer);
        ssize_t written = write(serial_fd_, buffer, 25);
        (void)written;
            
        static int debug_cnt = 0;
        if(debug_cnt++ % 50 == 0) 
        {
            std::cout << "[SBUS Hex]: ";
            for(int i = 0; i < 25; i++) 
            {
                printf("%02X ", buffer[i]); 
            }
            std::cout << std::endl;
        }
    }
}

void DuctRobotMove::moveshutdown()
{
    if(serial_fd_ != -1)
    {
        ch_[0] = 1000;
        ch_[1] = 1000;
        ch_[2] = 1000;
        ch_[3] = 500; 
        ch_[4] = 500;
        ch_[5] = 500;
        for(int i=0; i<5; i++) {
            send_loop();
            usleep(20000);
        }

        close(serial_fd_);
        serial_fd_ = -1;
        std::cout << "[DuctRobot] 涵道机器人已停止、步进电机脱机" << std::endl;
    }
}

void DuctRobotMove::pack_protocol_data(std::vector<uint16_t> ch, uint8_t* buf)
{
    if (ch.size() != 16) 
    {
        std::cerr << "需要 16 个通道数据" << std::endl;
        return;
    }

    for (int i = 0; i < 16; ++i) 
    {
        if(ch[i] > 1800) ch[i] = 1800;
        ch[i] &= 0x07FF;
    }

    buf[0] = 0x0F;

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

    buf[23] = 0x00;
    buf[24] = 0x00; 
}

