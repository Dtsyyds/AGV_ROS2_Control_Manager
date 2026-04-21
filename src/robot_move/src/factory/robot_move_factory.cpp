#include "robot_move/factory/robot_move_factory.hpp"

#include "robot_move/impl/agv/agv_robot_move.hpp"
#include "robot_move/impl/air/air_robot_move.hpp"
#include "robot_move/impl/duct/duct_robot_move.hpp"
#include "robot_move/impl/mag/mag_robot_move.hpp"

namespace robot_move {

std::unique_ptr<RobotMove> createRobotMove(const std::string& type)
{
    if (type == "air") 
    {
        return std::make_unique<AirRobotMove>();
    }
    if (type == "agv") 
    {
        return std::make_unique<AGVRobotMove>();
    }
    if (type == "mag") 
    {
        return std::make_unique<MagRobotMove>();
    }
    if (type == "duct") 
    {
        return std::make_unique<DuctRobotMove>();
    }
    return nullptr;
}

std::string defaultConnectionForType(const std::string& type)
{
    if (type == "air") return "/dev/ttyCH341USB0";
    if (type == "mag") return "/dev/ttyUSB1";
    if (type == "agv") return "10.42.0.114";
    if (type == "duct") return "/dev/ttyUSB1";
    return "";
}

}
