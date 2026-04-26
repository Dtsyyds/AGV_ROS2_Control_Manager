#include "robot_move/impl/mag/mag_robot_move.hpp"

#include <asm/termbits.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

MagRobotMove::MagRobotMove()
    : serial_fd_(-1),
      ch_(16, 1002),
      VAL_LOW(362),
      VAL_MID(1002),
      VAL_HIGH(1642)
{
}

bool MagRobotMove::init(const std::string& port)
{
    std::cout << "[MagRobot] 正在打开串口: " << port << " (115200, 8E1)..." << std::endl;
    serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if(serial_fd_ == -1)
    {
        std::cerr << "[MagRobot] 串口打开失败!" << std::endl;
        return false;
    }
        
    struct termios2 options;
    if (ioctl(serial_fd_, TCGETS2, &options) < 0) {
        std::cerr << "[MagRobot] 无法获取串口配置 (TCGETS2)" << std::endl;
        close(serial_fd_);
        return false;
    }

    options.c_cflag &= ~CBAUD;
    options.c_cflag |= B115200;
    options.c_ispeed = 115200;
    options.c_ospeed = 115200;

    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;

    options.c_cflag &= ~PARENB;  
    options.c_iflag &= ~(INPCK | ISTRIP); 

    options.c_cflag &= ~CSTOPB;

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    if (ioctl(serial_fd_, TCSETS2, &options) < 0) {
        std::cerr << "[MagRobot] 无法应用串口配置 (TCSETS2)" << std::endl;
        close(serial_fd_);
        return false;
    }

    ch_[6] = VAL_HIGH;
    ch_[7] = VAL_LOW; 
    std::cout << "[MagRobot] 发送解锁信号 (CH7 -> High)..." << std::endl;

    for(int i=0; i<20; i++) {
        send_loop();
        usleep(14000);
    }
    std::cout << "[MagRobot] 初始化完成: 电机已解锁 (CH7=High), 速度设为低速 (CH8=Low)" << std::endl;
    return true;
}

void MagRobotMove::setVelocity(double linear_x, double vy, double angle_z)
{
    (void)vy;

    if (linear_x >= 0.01)
    {
        ch_[1] = VAL_HIGH;
    }
    else if (linear_x <= -0.01)
    {
        ch_[1] = VAL_LOW;
    }
    else
    {
        ch_[1] = VAL_MID;
    }

    if (angle_z >= 0.01)
    {
        ch_[0] = VAL_LOW;
    }
    else if (angle_z <= -0.01)
    {
        ch_[0] = VAL_HIGH;
    }
    else
    {
        ch_[0] = VAL_MID;
    }
}

void MagRobotMove::send_loop()
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

void MagRobotMove::moveshutdown()
{
    if(serial_fd_ != -1)
    {
        std::fill(ch_.begin(), ch_.end(), VAL_MID);
        ch_[6] = VAL_LOW;
            
        for(int i=0; i<5; i++) {
            send_loop();
            usleep(20000);
        }
            
        close(serial_fd_);
        serial_fd_ = -1;
        std::cout << "[MagRobot] 已停止并关闭串口。" << std::endl;
    }
}

void MagRobotMove::pack_protocol_data(const std::vector<uint16_t>& ch, uint8_t* buf)
{
    if (ch.size() != 16) return;

    std::vector<uint16_t> clean_ch = ch;
    for(auto& val : clean_ch) {
        if(val > 2047) val = 2047;
    }

    buf[0] = 0x0F;

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

    buf[23] = 0x00;
    buf[24] = 0x00;
}

