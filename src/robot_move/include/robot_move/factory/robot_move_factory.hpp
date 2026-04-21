#ifndef ROBOT_MOVE_FACTORY_HPP
#define ROBOT_MOVE_FACTORY_HPP

#include <memory>
#include <string>

#include "robot_move/interface/robot_move_base.hpp"

namespace robot_move {

std::unique_ptr<RobotMove> createRobotMove(const std::string& type);
std::string defaultConnectionForType(const std::string& type);

}

#endif
