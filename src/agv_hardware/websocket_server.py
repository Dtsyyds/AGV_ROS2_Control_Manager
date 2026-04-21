#!/usr/bin/env python3
"""
WebSocket服务器，用于接收和发送AGV机械臂控制指令
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
import threading
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AGVWebsocketServer:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port
        self.clients = set()
        self.running = False
        
        # 机械臂关节位置（模拟）
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    async def register_client(self, websocket):
        """注册新客户端"""
        self.clients.add(websocket)
        logger.info(f"新客户端连接: {websocket.remote_address}")
        
        # 发送连接确认
        connection_msg = {
            "type": "connection_ack",
            "server": "agv_control_server",
            "timestamp": str(int(datetime.now().timestamp() * 1000))
        }
        await self.send_to_client(websocket, connection_msg)
    
    async def unregister_client(self, websocket):
        """注销客户端"""
        self.clients.discard(websocket)
        logger.info(f"客户端断开: {websocket.remote_address}")
    
    async def send_to_client(self, websocket, message):
        """向特定客户端发送消息"""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"客户端连接已关闭，无法发送消息: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    
    async def broadcast_to_clients(self, message):
        """广播消息给所有客户端"""
        if not self.clients:
            return
            
        # 移除已断开的连接
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(client)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected_clients.append(client)
        
        # 从客户端集合中移除已断开的连接
        for client in disconnected_clients:
            self.clients.discard(client)
    
    def parse_and_execute_command(self, message_json):
        """解析并执行命令"""
        try:
            data = json.loads(message_json)
            command_type = data.get("type", "")
            
            if command_type == "connection":
                logger.info(f"收到连接确认: {data}")
                return {"status": "connected", "message": "Server connected successfully"}
            
            elif command_type == "joint_control":
                joint_data = data.get("data", {})
                joints = [
                    joint_data.get("joint1", self.joint_positions[0]),
                    joint_data.get("joint2", self.joint_positions[1]),
                    joint_data.get("joint3", self.joint_positions[2]),
                    joint_data.get("joint4", self.joint_positions[3]),
                    joint_data.get("joint5", self.joint_positions[4]),
                    joint_data.get("joint6", self.joint_positions[5])
                ]
                
                self.joint_positions = joints
                logger.info(f"执行关节控制命令: {joints}")
                
                # 返回确认消息
                return {
                    "type": "command_ack",
                    "command": "joint_control",
                    "status": "executed",
                    "positions": joints
                }
            
            elif command_type == "move_to_position":
                position_data = data.get("data", {})
                joints = [
                    position_data.get("j1", self.joint_positions[0]),
                    position_data.get("j2", self.joint_positions[1]),
                    position_data.get("j3", self.joint_positions[2]),
                    position_data.get("j4", self.joint_positions[3]),
                    position_data.get("j5", self.joint_positions[4]),
                    position_data.get("j6", self.joint_positions[5])
                ]
                
                self.joint_positions = joints
                logger.info(f"执行移动到位置命令: {joints}")
                
                # 返回确认消息
                return {
                    "type": "command_ack", 
                    "command": "move_to_position",
                    "status": "executed",
                    "positions": joints
                }
            
            elif command_type == "emergency_stop":
                logger.warning("执行紧急停止命令")
                # 紧急停止逻辑可以在这里实现
                return {
                    "type": "command_ack",
                    "command": "emergency_stop", 
                    "status": "executed"
                }
            
            elif command_type == "power_on":
                logger.info("执行上电命令")
                return {
                    "type": "command_ack",
                    "command": "power_on",
                    "status": "executed"
                }
            
            elif command_type == "power_off":
                logger.info("执行断电命令")
                return {
                    "type": "command_ack", 
                    "command": "power_off",
                    "status": "executed"
                }
            
            elif command_type == "get_status":
                # 返回当前状态
                return {
                    "type": "status_response",
                    "joint_positions": self.joint_positions,
                    "timestamp": str(int(datetime.now().timestamp() * 1000))
                }
            
            else:
                logger.warning(f"未知命令类型: {command_type}")
                return {
                    "type": "error",
                    "message": f"Unknown command type: {command_type}"
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            return {
                "type": "error",
                "message": f"Invalid JSON: {str(e)}"
            }
        except Exception as e:
            logger.error(f"执行命令时出错: {e}")
            return {
                "type": "error", 
                "message": f"Command execution error: {str(e)}"
            }
    
    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                logger.info(f"收到客户端消息: {message}")
                
                # 解析并执行命令
                response = self.parse_and_execute_command(message)
                
                # 发送响应
                if response:
                    await self.send_to_client(websocket, response)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端连接已关闭: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"处理客户端消息时出错: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def start_server(self):
        """启动WebSocket服务器"""
        self.running = True
        logger.info(f"启动WebSocket服务器在 {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info(f"WebSocket服务器已在 {self.host}:{self.port} 上运行")
            
            # 保持服务器运行
            while self.running:
                await asyncio.sleep(1)
    
    def stop_server(self):
        """停止服务器"""
        self.running = False
        logger.info("正在停止WebSocket服务器...")
    
    def send_joint_positions_to_all(self):
        """向所有客户端发送当前关节位置"""
        status_msg = {
            "type": "current_positions",
            "data": {
                "joint1": self.joint_positions[0],
                "joint2": self.joint_positions[1], 
                "joint3": self.joint_positions[2],
                "joint4": self.joint_positions[3],
                "joint5": self.joint_positions[4],
                "joint6": self.joint_positions[5]
            },
            "timestamp": str(int(datetime.now().timestamp() * 1000))
        }
        
        # 在事件循环中发送消息
        async def send():
            await self.broadcast_to_clients(status_msg)
        
        # 获取当前事件循环并运行发送任务
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(send(), loop)
            else:
                loop.run_until_complete(send())
        except RuntimeError:
            # 如果没有运行的事件循环，则创建一个新的
            asyncio.run(send())


def run_server():
    """运行服务器的便捷函数"""
    server = AGVWebsocketServer(host="0.0.0.0", port=9000)
    
    def periodic_status_update():
        """定期发送状态更新"""
        while server.running:
            time.sleep(5)  # 每5秒发送一次状态
            if server.clients:  # 只有在有客户端连接时才发送
                server.send_joint_positions_to_all()
    
    # 启动状态更新线程
    status_thread = threading.Thread(target=periodic_status_update, daemon=True)
    
    try:
        status_thread.start()
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
        server.stop_server()
    finally:
        logger.info("服务器已关闭")


if __name__ == "__main__":
    print("AGV WebSocket 服务器")
    print("服务器将监听 ws://0.0.0.0:9000")
    print("按 Ctrl+C 停止服务器")
    
    run_server()