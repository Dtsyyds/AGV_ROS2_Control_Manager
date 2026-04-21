#ifndef AIR_ROBOT_MOVE_HPP
#define AIR_ROBOT_MOVE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "robot_move/interface/robot_move_base.hpp"

class AirRobotMove : public RobotMove
{
private:
    int serial_fd_;
    std::vector<uint16_t> ch_;

    void pack_protocol_data(std::vector<uint16_t> ch, uint8_t* buf);

public:
    AirRobotMove();

    bool init(const std::string& port) override;
    void setVelocity(double linear_x, double vy, double angle_z) override;
    void send_loop() override;
    void moveshutdown() override;

    const std::vector<uint16_t> BACKWARD;
    const std::vector<uint16_t> FORWARD;
    const std::vector<uint16_t> TURN_LEFT;
    const std::vector<uint16_t> TURN_RIGHT;
    const std::vector<uint16_t> STOP;
};

#endif
