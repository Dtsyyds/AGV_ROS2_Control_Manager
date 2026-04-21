#include "robot_move/impl/air/air_robot_move.hpp"

#include <asm/termbits.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cmath>
#include <iostream>

AirRobotMove::AirRobotMove()
    : serial_fd_(-1),
      ch_(16),
      BACKWARD({1500,1730,1500,1500,1,1,1,0, 0,0,1050,1950,0,0,0,0}),
      FORWARD({1500,1270,1500,1500,1,1,1,0, 0,0,1050,1950,0,0,0,0}),
      TURN_LEFT({1200,1500,1500,1500,1,1500,1,1500, 1500,1500,1050,1950,0,0,0,0}),
      TURN_RIGHT({1900,1500,1500,1500,1,1500,1,1500, 1500,1500,1050,1950,0,0,0,0}),
      STOP({1500,1500,1500,1500,1,1,1,0, 0,0,1050,1900,0,0,0,0})
{
    for(int i = 0; i < 16; i++)
    {
        ch_[i] = STOP.at(i);
    }
}

bool AirRobotMove::init(const std::string& port)
{
    serial_fd_ = open(port.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if(serial_fd_ == -1)
    {
        std::cerr << "串口打开失败" << std::endl;
        return false;
    }
        
    struct termios2 options;
    if (ioctl(serial_fd_, TCGETS2, &options) < 0) {
        std::cerr << "[AirRobot] 无法获取串口配置 (TCGETS2)" << std::endl;
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
    options.c_iflag |= (INPCK | ISTRIP);

    options.c_cflag |= CSTOPB;

    options.c_cflag |= (CLOCAL | CREAD);
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_oflag &= ~OPOST;
    options.c_iflag &= ~(IXON | IXOFF | IXANY);

    if (ioctl(serial_fd_, TCSETS2, &options) < 0) {
        std::cerr << "[AirRobot] 无法应用串口配置 (TCSETS2)" << std::endl;
        close(serial_fd_);
        return false;
    }

    return true;
}

void AirRobotMove::setVelocity(double linear_x, double vy, double angle_z)
{
    (void)vy;
        
    const double LINEAR_SCALE = 23000.0;
        
    if(linear_x > 0.001)
    {
        int val = 1500 - (int)(linear_x * LINEAR_SCALE);
        if(val <= 1050) val = 1050; 
        ch_[1] = (uint16_t)val;
    }
    else if(linear_x < -0.001)
    {
        int val = 1500 + (int)(std::abs(linear_x) * LINEAR_SCALE);
        if(val >= 1950) val = 1950;
        ch_[1] = (uint16_t)val;
    }
    else
    {
        ch_[1] = 1500;
    }

    if(angle_z > 0.001)
    {
        const double LEFT_SCALE = 30000.0;
        int val = 1500 - (int)(angle_z * LEFT_SCALE);
            
        if(val <= 1050) val = 1050;
        ch_[0] = (uint16_t)val;
    }
    else if(angle_z < -0.001)
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

void AirRobotMove::send_loop()
{
    if(serial_fd_ != -1)
    {
        uint8_t buffer[25];
        pack_protocol_data(ch_, buffer);
        write(serial_fd_, buffer, 25);
    }
}

void AirRobotMove::moveshutdown()
{
    if(serial_fd_ != -1)
    {
        close(serial_fd_);
    }
}

void AirRobotMove::pack_protocol_data(std::vector<uint16_t> ch, uint8_t* buf)
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

    buf[0] = 0x0F;

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

    buf[23] = 0x00;

    unsigned int sum = 0;
    for(int i=0; i<24; i++) 
    {
        sum += buf[i];
    }
    buf[24] = (uint8_t)(sum & 0xFF);
}
