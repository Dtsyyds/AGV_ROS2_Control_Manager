#ifndef DUCT_ROBOT_MOVE_HPP
#define DUCT_ROBOT_MOVE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "robot_move/interface/robot_move_base.hpp"

class DuctRobotMove : public RobotMove
{
private:
    int serial_fd_;
    std::vector<uint16_t> ch_;

    void pack_protocol_data(std::vector<uint16_t> ch, uint8_t* buf);

public:
    DuctRobotMove();

    bool init(const std::string& port) override;
    void setVelocity(double linear_x, double vy, double angle_z) override;
    void setScannerConfig(double speed, double distance, double precision) override;
    void scannercontrol() override;
    void send_loop() override;
    void moveshutdown() override;
};

#endif
