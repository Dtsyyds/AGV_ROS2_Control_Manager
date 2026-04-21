#ifndef ROBOT_MOVE_BASE_HPP
#define ROBOT_MOVE_BASE_HPP
 
#include <string>
 
// ==========================================
// 1. 抽象基类 (Base Class)
// ==========================================
class RobotMove 
{
private:
 
public:
    RobotMove() = default;
 
    virtual bool init(const std::string& connection_str) = 0;//初始化硬件连接（TCP/SBUS）
 
    virtual void setVelocity(double vx, double vy, double wz) = 0;
 
    virtual void send_loop() = 0;
 
    virtual void moveshutdown() = 0;
 
    /* -----------------仅气吸附和涵道控制方法-----------------*/
    virtual void setScannerConfig(double speed, double distance, double precision){};
 
    virtual void scannercontrol(){};
 
    /* -----------------AGV 独有控制方法-----------------*/
    // 特殊动作接口 (仅 AGV 支持, 气吸附返回 false)
    virtual bool movebydistance(double dist, double vx, double vy) { return false; }
    virtual bool rotatebyangle(double angle, double vw) { return false; }
    // 获取位置接口 (x, y, theta)
    virtual bool getpose(double& x, double& y, double& theta) { return false; }
 
 
 
    // virtual ~RobotMove() = default;
};
 
#endif
