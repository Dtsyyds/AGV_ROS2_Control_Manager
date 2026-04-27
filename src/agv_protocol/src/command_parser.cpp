#include "agv_protocol/command_parser.hpp"
#include <iostream>
#include <algorithm>

namespace agv_protocol {

std::string CommandParser::safeGetString(const Json::Value& value, const std::string& key, const std::string& default_val) {
    return value.isMember(key) && value[key].isString() ? value[key].asString() : default_val;
}

double CommandParser::safeGetDouble(const Json::Value& value, const std::string& key, double default_val) {
    return value.isMember(key) && value[key].isNumeric() ? value[key].asDouble() : default_val;
}

Json::Value CommandParser::createBaseMessage(const std::string& robot_id, const std::string& msg_type) {
    Json::Value root;
    root["header"]["robot_id"] = robot_id;
    root["header"]["msg_type"] = msg_type;
    return root;
}

std::string CommandParser::buildGenericMessage(const std::string& robot_id,
                                                const std::string& msg_type,
                                                const Json::Value& payload)
{
    Json::Value root = createBaseMessage(robot_id, msg_type);
    root["payload"] = payload;

    Json::StreamWriterBuilder writer;
    return Json::writeString(writer, root);
}

std::optional<CommandData> CommandParser::parse(const std::string& json_str)
{
    try {
        Json::Value root;
        Json::CharReaderBuilder reader;
        std::string errors;
        std::stringstream ss(json_str);

        if (!Json::parseFromStream(reader, ss, &root, &errors)) {
            std::cerr << "[CommandParser] JSON 解析错误：" << errors << std::endl;
            return std::nullopt;
        }

        CommandData cmd;

        if (!parseHeader(root, cmd)) {
            return std::nullopt;
        }

        Json::Value payload = root["payload"];
        if (!payload.isObject()) {
            std::cerr << "[CommandParser] 缺少 payload 字段" << std::endl;
            return std::nullopt;
        }

        if (!parsePayload(payload, cmd)) {
            return std::nullopt;
        }

        return cmd;
    }
    catch (const std::exception& e) {
        std::cerr << "[CommandParser] 解析命令时发生异常：" << e.what() << std::endl;
        return std::nullopt;
    }
}

bool CommandParser::parseHeader(const Json::Value& root, CommandData& cmd)
{
    const Json::Value& header = root["header"];
    if (!header.isObject()) {
        std::cerr << "[CommandParser] 缺少 header 字段" << std::endl;
        return false;
    }

    cmd.header.robot_id = safeGetString(header, "robot_id", "mobile_dual_arm_robot");
    cmd.header.msg_type = safeGetString(header, "msg_type", "command");

    return true;
}

bool CommandParser::parsePayload(const Json::Value& payload, CommandData& cmd)
{
    // 1. 处理 check 类型的消息
    if (cmd.header.msg_type == "check") {
        for (const auto& key : payload.getMemberNames()) {
            const Json::Value& device_payload = payload[key];
            if (device_payload.isObject() && 
                device_payload.isMember("command") && 
                device_payload["command"].asString() == "check") 
            {
                cmd.check_devices.push_back(key);
            }
        }
        return true; 
    }

    // 2. 解析机械臂命令
    if (payload.isMember("arms")) {
        parseArmCommands(payload["arms"], cmd.arm_commands);
    }

    // 3. 解析底盘命令 (支持 "chassis" 或 "agv" 格式)
    if (payload.isMember("chassis")) {
        parseChassisCommand(payload["chassis"], cmd);
    } else if (payload.isMember("agv")) {
        parseAgvCommand(payload["agv"], cmd);
    }

    // 4. 解析相机命令
    if (payload.isMember("camera")) {
        parseCameraCommand(payload["camera"], cmd);
    }

    return true;
}

bool CommandParser::parseCameraCommand(const Json::Value& camera, CommandData& cmd)
{
    if (!camera.isObject()) return false;

    cmd.camera_cmd.command = safeGetString(camera, "command");
    cmd.camera_cmd.x = safeGetDouble(camera, "x");
    cmd.camera_cmd.y = safeGetDouble(camera, "y");
    cmd.camera_cmd.scan_mode = safeGetString(camera, "scan_mode", "Long");
    cmd.camera_cmd.spacing = (int)safeGetDouble(camera, "spacing");
    cmd.camera_cmd.shrink_factor = (int)safeGetDouble(camera, "shrink_factor");
    cmd.camera_cmd.default_InterPoints = (int)safeGetDouble(camera, "default_InterPoints");
    
    cmd.has_camera_command = true;
    return true;
}

bool CommandParser::parseArmCommands(const Json::Value& arms, std::vector<ArmCommand>& commands)
{
    if (!arms.isArray()) return false;

    static const std::array<std::string, 6> JOINT_KEYS = {"val1", "val2", "val3", "val4", "val5", "val6"};
    static const std::array<std::string, 6> CARTESIAN_KEYS = {"x", "y", "z", "roll", "pitch", "yaw"};

    for (Json::ArrayIndex i = 0; i < arms.size(); ++i) {
        const Json::Value& arm = arms[i];
        if (!arm.isObject()) continue;

        ArmCommand cmd;
        cmd.arm_id = safeGetString(arm, "arm_id", "left_arm");
        cmd.command = safeGetString(arm, "command", "joint_move");

        const Json::Value& params = arm["parameters"];
        if (params.isObject()) {
            if (params.isMember("val1")) {
                cmd.is_joint_command = true;
                extractToVector<6>(params, JOINT_KEYS, cmd.parameters);
            } else {
                cmd.is_joint_command = false;
                extractToVector<6>(params, CARTESIAN_KEYS, cmd.parameters);
            }
            cmd.speed = safeGetDouble(params, "speed", 50.0);
        }
        commands.push_back(cmd);
    }
    return true;
}

bool CommandParser::parseChassisCommand(const Json::Value& chassis, CommandData& cmd)
{
    std::string command = safeGetString(chassis, "command");
    const Json::Value& params = chassis["parameters"];

    if (command == "emergency_stop") {
        cmd.emergency_stop = true;
    } else if (command == "velocity_move") {
        if (params.isObject()) {
            cmd.chassis_velocity[0] = params.isMember("vx") ? safeGetDouble(params, "vx") : safeGetDouble(params, "x");
            cmd.chassis_velocity[1] = params.isMember("vy") ? safeGetDouble(params, "vy") : safeGetDouble(params, "y");
            cmd.chassis_velocity[2] = params.isMember("wz") ? safeGetDouble(params, "wz") : safeGetDouble(params, "yaw");
            cmd.has_chassis_command = true;
        }
    }
    return true;
}

bool CommandParser::parseAgvCommand(const Json::Value& agv, CommandData& cmd)
{
    if (!agv.isObject()) return false;
    
    std::string command = safeGetString(agv, "command");
    const Json::Value& data = agv["data"];
    
    if (command == "move" && data.isObject()) {
        // 解析速度参数
        cmd.chassis_velocity[0] = data.isMember("vx") ? safeGetDouble(data, "vx") : 0.0;
        cmd.chassis_velocity[1] = data.isMember("vy") ? safeGetDouble(data, "vy") : 0.0;
        cmd.chassis_velocity[2] = data.isMember("wz") ? safeGetDouble(data, "wz") : 0.0;
        cmd.has_chassis_command = true;
        std::cout << "[CommandParser] 解析AGV命令: vx=" << cmd.chassis_velocity[0] 
                  << ", vy=" << cmd.chassis_velocity[1] 
                  << ", wz=" << cmd.chassis_velocity[2] << std::endl;
    }
    return true;
}

std::string CommandParser::buildStatusMessage(const std::array<double, 6>& joint_positions,
                                             const std::string& robot_id)
{
    Json::Value payload;
    payload["arms"][0]["arm_id"] = "left_arm";
    payload["arms"][0]["pos_type"] = "joint";
    for(size_t i=0; i<6; ++i) payload["arms"][0]["joint" + std::to_string(i+1)] = joint_positions[i];

    payload["arms"][1]["arm_id"] = "right_arm";
    payload["arms"][1]["pos_type"] = "joint";
    for(size_t i=1; i<=6; ++i) payload["arms"][1]["joint" + std::to_string(i)] = 0.0;

    return buildGenericMessage(robot_id, "status_update", payload);
}

std::string CommandParser::buildPosturePositionsMessage(const std::array<double, 6>& posture_positions,
                                                       const std::string& robot_id)
{
    Json::Value payload;
    payload["arms"][0]["arm_id"] = "left_arm";
    payload["arms"][0]["pos_type"] = "cartesian";
    payload["arms"][0]["x"] = posture_positions[0];
    payload["arms"][0]["y"] = posture_positions[1];
    payload["arms"][0]["z"] = posture_positions[2];
    payload["arms"][0]["rx"] = posture_positions[3];
    payload["arms"][0]["ry"] = posture_positions[4];
    payload["arms"][0]["rz"] = posture_positions[5];

    payload["arms"][1]["arm_id"] = "right_arm";
    payload["arms"][1]["pos_type"] = "cartesian";
    payload["arms"][1]["x"] = 0.0; payload["arms"][1]["y"] = 0.0; payload["arms"][1]["z"] = 0.0;
    payload["arms"][1]["rx"] = 0.0; payload["arms"][1]["ry"] = 0.0; payload["arms"][1]["rz"] = 0.0;

    return buildGenericMessage(robot_id, "status_update", payload);
}

std::string CommandParser::buildCommandResponse(const std::string& arm_id,
                                                 const std::string& command,
                                                 bool success)
{
    Json::Value payload;
    payload["arms"][0]["arm_id"] = arm_id;
    payload["arms"][0]["command"] = command;
    payload["arms"][0]["result"] = success ? (command + "_successed") : "failed";

    return buildGenericMessage("mobile_dual_arm_robot", "command", payload);
}

std::string CommandParser::buildCheckResponse(const std::vector<std::pair<std::string, bool>>& devices)
{
    Json::Value payload;
    for (const auto& device : devices) {
        Json::Value device_node;
        device_node["command"] = "check";
        device_node["result"] = device.second ? "yes" : "no";
        payload[device.first] = device_node;
    }
    return buildGenericMessage("mobile_dual_arm_robot", "response", payload);
}

} // namespace agv_protocol
