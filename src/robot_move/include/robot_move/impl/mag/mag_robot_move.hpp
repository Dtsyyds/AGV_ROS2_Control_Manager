#ifndef MAG_ROBOT_MOVE_HPP
#define MAG_ROBOT_MOVE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "robot_move/interface/robot_move_base.hpp"

class MagRobotMove : public RobotMove
{
private:
    int serial_fd_;
    std::vector<uint16_t> ch_;

    void pack_protocol_data(const std::vector<uint16_t>& ch, uint8_t* buf);

    const uint16_t VAL_LOW;
    const uint16_t VAL_MID;
    const uint16_t VAL_HIGH;

public:
    MagRobotMove();

    bool init(const std::string& port) override;
    void setVelocity(double linear_x, double vy, double angle_z) override;
    void send_loop() override;
    void moveshutdown() override;
};

#endif
