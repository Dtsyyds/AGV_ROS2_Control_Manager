#ifndef AGV_ROBOT_MOVE_HPP
#define AGV_ROBOT_MOVE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "robot_move/interface/robot_move_base.hpp"

#ifndef SEERCONTROLIP
#define SEERCONTROLIP "10.42.0.114"//AGV IP
#endif

class AGVRobotMove : public RobotMove
{
private:
    int sock_19204_;
    int sock_19205_;
    int sock_19206_;
    int sock_19210_;
    uint16_t seq_num_;

    bool connectPort(const std::string& ip, int port, int& sock_fd);
    void unlockBrake();
    std::vector<uint8_t> packSeerFrame(uint16_t type, const std::string& json);
    void readresponse(int sock_fd, const std::string& cmd_name);
    bool getcurrentpose(double& x, double& y, double& theta);
    void robot_control_reloc_req();
    bool robot_navigation_status();

public:
    AGVRobotMove();

    bool init(const std::string& seer_ip) override;
    void setVelocity(double vx, double vy, double wz) override;
    void send_loop() override;
    void moveshutdown() override;

    bool movebydistance(double dist, double vx, double vy) override;
    bool rotatebyangle(double angle, double vw) override;
    bool getpose(double& x, double& y, double& theta) override;
};

#endif
