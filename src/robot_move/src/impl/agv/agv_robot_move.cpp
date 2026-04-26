#include "robot_move/impl/agv/agv_robot_move.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "robot_move/interface/cJSON.h"

AGVRobotMove::AGVRobotMove()
    : sock_19204_(-1),
      sock_19205_(-1),
      sock_19206_(-1),
      sock_19210_(-1),
      seq_num_(0)
{
}

bool AGVRobotMove::connectPort(const std::string& ip, int port, int& sock_fd)
{
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) 
    {
        return false;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) <= 0)
    {
        return false;
    }

    struct timeval timeout;      
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout,sizeof(timeout));
    setsockopt(sock_fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout,sizeof(timeout));

    if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0)
    {
        close(sock_fd);
        sock_fd = -1;
        return false;
    }
    return true;
}

std::vector<uint8_t> AGVRobotMove::packSeerFrame(uint16_t type, const std::string& json)
{
    uint32_t len = json.length();
    std::vector<uint8_t> frame(16 + len);

    frame[0] = 0x5A;
    frame[1] = 0x01;
        
    seq_num_++;
    frame[2] = (seq_num_ >> 8) & 0xFF; 
    frame[3] = seq_num_ & 0xFF;

    frame[4] = (len >> 24) & 0xFF;
    frame[5] = (len >> 16) & 0xFF;
    frame[6] = (len >> 8) & 0xFF;
    frame[7] = len & 0xFF;

    frame[8] = (type >> 8) & 0xFF;
    frame[9] = type & 0xFF;

    memset(&frame[10], 0, 6);
    memcpy(&frame[16], json.c_str(), len);
    return frame;
}

void AGVRobotMove::readresponse(int sock_fd, const std::string& cmd_name)
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

bool AGVRobotMove::getcurrentpose(double& x, double& y, double& theta)
{
    if (sock_19204_ < 0) return false;
    std::vector<uint8_t> frame = packSeerFrame(1004, "{}");
    send(sock_19204_, frame.data(), frame.size(), 0);

    char buffer[4096]; 
    memset(buffer, 0, sizeof(buffer));
    int len = recv(sock_19204_, buffer, sizeof(buffer)-1, 0);
    if(len > 16)
    {
        char *resp_data = buffer + 16;
        cJSON *json_data = cJSON_Parse(resp_data);
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

void AGVRobotMove::robot_control_reloc_req()
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
    cJSON_AddNumberToObject(root, "length", 2);
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

bool AGVRobotMove::robot_navigation_status()
{
    if(sock_19204_ < 0) return false;
    std::cout << "[AGV] 等待导航任务完成..." << std::endl;
    while(true)
    {
        cJSON *navi_root = cJSON_CreateObject();
        cJSON_AddBoolToObject(navi_root, "simple", true);
        char* str = cJSON_PrintUnformatted(navi_root);
        if(str)
        {
            std::vector<uint8_t> navi_frame = packSeerFrame(1020, str);
            send(sock_19204_, navi_frame.data(), navi_frame.size(), 0);
            free(str);
        }
        cJSON_Delete(navi_root);

        char buf[4096];
        int len = recv(sock_19204_, buf, 4095, 0);
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
                if (status == 0 || status == 4)
                {
                    cJSON_Delete(resp_data);
                    std::cout << "[AGV] 导航完成 (Status: " << status << ")" << std::endl;
                    return true;
                }
                if (status == 5) 
                {
                    cJSON_Delete(resp_data);
                    std::cerr << "[AGV] 导航失败 (Status: 5)!" << std::endl;
                    return false;
                }
            }
            cJSON_Delete(resp_data);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void AGVRobotMove::unlockBrake()
{
    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "id", 0);
    cJSON_AddBoolToObject(root, "status", true);
    char* json_str = cJSON_PrintUnformatted(root);
    if(json_str)
    {
        std::vector<uint8_t> frame = packSeerFrame(6001, json_str);
        if (sock_19210_ >= 0) 
        {
            send(sock_19210_, frame.data(), frame.size(), 0);
        }
        free(json_str);
    }
    cJSON_Delete(root);
}

bool AGVRobotMove::init(const std::string& seer_ip)
{
    std::string ip = seer_ip;
    size_t colon = seer_ip.find(':');
    if (colon != std::string::npos) 
    {
        ip = seer_ip.substr(0, colon);
    }
    std::cout << "[AGV] 初始化连接到 " << ip << "..." << std::endl;

    bool s10 = connectPort(ip, 19210, sock_19210_);
    bool s04 = connectPort(ip, 19204, sock_19204_);
    bool s05 = connectPort(ip, 19205, sock_19205_);
    bool s06 = connectPort(ip, 19206, sock_19206_);

    if (s10) 
    {
        std::cout << "[AGV]连接19210成功..." << std::endl;
    }
    else
    {
        std::cerr << "[AGV]连接19210失败..." << std::endl;
    }
    if (s04) 
    {
        std::cout << "[AGV]连接19204成功..." << std::endl;
    } 
    else 
    {
        std::cerr << "[AGV]连接19204失败..." << std::endl;
    }
    if (s05) 
    {
        std::cout << "[AGV]连接19205成功..." << std::endl;
    } 
    else 
    {
        std::cerr << "[AGV]连接19205失败..." << std::endl;
        return false;
    }
    if (s06) 
    {
        std::cout << "[AGV]连接19206成功..." << std::endl;
    } 
    else 
    {
        std::cerr << "[AGV]连接19206失败..." << std::endl;
    }
    unlockBrake();
    std::cout << "[AGV] 初始化端口结束！" << std::endl;

    std::cout << "[AGV] 开始自动重定位..." << std::endl;
    robot_control_reloc_req();
    return true;
}

bool AGVRobotMove::getpose(double& x, double& y, double& theta)
{
    return getcurrentpose(x, y, theta);
}

void AGVRobotMove::setVelocity(double vx, double vy, double wz)
{
    if (sock_19205_ < 0) return;
    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "vx", vx);
    cJSON_AddNumberToObject(root, "vy", vy);
    cJSON_AddNumberToObject(root, "w", wz);
    cJSON_AddNumberToObject(root, "duration", 0);
    char* json_str = cJSON_PrintUnformatted(root);
        
    if(json_str)
    {
        std::vector<uint8_t> frame = packSeerFrame(2010, json_str);
        send(sock_19205_, frame.data(), frame.size(), 0);
        free(json_str);
    }

    cJSON_Delete(root);
}

bool AGVRobotMove::movebydistance(double dist, double vx, double vy)
{
    if(sock_19206_ < 0) return false; 
    std::cout << "[AGV] 直线运动距离: " << dist << "m" << std::endl;

    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "dist", dist);
    cJSON_AddNumberToObject(root, "vx", vx);
    cJSON_AddNumberToObject(root, "vy", vy);
    char* json_str = cJSON_PrintUnformatted(root);
    if(json_str)
    {
        std::vector<uint8_t> frame = packSeerFrame(3055, json_str);
        send(sock_19206_, frame.data(), frame.size(), 0);
        free(json_str);
        readresponse(sock_19206_, "平动");
        return robot_navigation_status();
    }
    cJSON_Delete(root);
    return false;
}

bool AGVRobotMove::rotatebyangle(double angle, double vw)
{
    if(sock_19206_ < 0) return false; 
    std::cout << "[AGV] 转动角度: " << angle << "rad" << std::endl;

    cJSON *root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "angle", angle);
    cJSON_AddNumberToObject(root, "vw", vw);
    char* json_str = cJSON_PrintUnformatted(root);
    if(json_str)
    {
        std::vector<uint8_t> frame = packSeerFrame(3056, json_str);
        send(sock_19206_, frame.data(), frame.size(), 0);
        free(json_str);
        readresponse(sock_19206_, "转动");
        return robot_navigation_status();
    }
    cJSON_Delete(root);
    return false;
}

void AGVRobotMove::send_loop()
{
}

void AGVRobotMove::moveshutdown()
{
    if (sock_19204_ >= 0) close(sock_19204_);
    if (sock_19205_ >= 0) close(sock_19205_);
    if (sock_19206_ >= 0) close(sock_19206_);
    if (sock_19210_ >= 0) close(sock_19210_);

    std::cout << "AGV 已停止运行！！！" << std::endl;
}

