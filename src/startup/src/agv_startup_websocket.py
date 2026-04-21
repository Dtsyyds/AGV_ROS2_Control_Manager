#!/usr/bin/env python3
"""
WebSocket服务器，用于接收启动AGV的指令
"""

import asyncio
import websockets
import json
import logging
import subprocess
import os
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AGVStartupWebsocketServer:
    def __init__(self, host="0.0.0.0", port=9001):
        self.host = host
        self.port = port
        self.clients = set()
        self.running = False
        
        # 记录启动的进程
        self.processes = {}
        
        # 线程池用于执行阻塞操作
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def register_client(self, websocket):
        """注册新客户端"""
        self.clients.add(websocket)
        logger.info(f"新客户端连接: {websocket.remote_address}")

        # 发送连接确认
        connection_msg = {
            "type": "connection_ack",
            "server": "agv_startup_server",
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

    def execute_launch_command(self, target, command):
        """执行启动或停止launch文件的命令"""
        try:
            if target == "arm_left_start":
                launch_file = "agv.launch.xml"
                
                if command.lower() == "true":
                    # 启动agv.launch.xml
                    if target in self.processes and self.processes[target].poll() is None:
                        logger.info(f"{target} 已经在运行中")
                        return {
                            "type": "command_ack",
                            "target": target,
                            "status": "already_running",
                            "message": f"{target} is already running"
                        }
                    
                    cmd = ["ros2", "launch", "agv_bringup", launch_file]
                    logger.info(f"启动命令: {' '.join(cmd)}")
                    
                    # 使用subprocess.Popen启动进程
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # 保存进程引用
                    self.processes[target] = process
                    
                    # 创建线程监控进程输出
                    def monitor_process():
                        while True:
                            output = process.stdout.readline()
                            if output == '' and process.poll() is not None:
                                break
                            if output:
                                logger.info(f"Process output: {output.strip()}")
                        
                        # 检查进程退出码
                        rc = process.poll()
                        if rc != 0:
                            error_output = process.stderr.read()
                            logger.error(f"Process exited with code {rc}: {error_output}")
                    
                    # 启动监控线程
                    monitor_thread = threading.Thread(target=monitor_process, daemon=True)
                    monitor_thread.start()
                    
                    return {
                        "type": "command_ack",
                        "target": target,
                        "status": "started",
                        "message": f"Started {target} with PID {process.pid}"
                    }
                    
                elif command.lower() == "false":
                    # 停止agv.launch.xml
                    if target in self.processes:
                        process = self.processes[target]
                        if process.poll() is None:  # 进程仍在运行
                            logger.info(f"停止进程 {target} (PID: {process.pid})")
                            
                            # 尝试优雅地终止进程
                            process.terminate()
                            
                            try:
                                # 等待进程结束，最多等待5秒
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                # 如果进程没有在5秒内结束，强制杀死
                                logger.warning(f"进程 {target} (PID: {process.pid}) 没有优雅退出，强制杀死")
                                process.kill()
                                process.wait()  # 等待进程完全结束
                            
                            del self.processes[target]
                            
                            return {
                                "type": "command_ack",
                                "target": target,
                                "status": "stopped",
                                "message": f"Stopped {target} (PID: {process.pid})"
                            }
                        else:
                            logger.info(f"{target} 进程已经停止")
                            return {
                                "type": "command_ack",
                                "target": target,
                                "status": "already_stopped",
                                "message": f"{target} is already stopped"
                            }
                    else:
                        logger.info(f"{target} 进程未找到")
                        return {
                            "type": "command_ack",
                            "target": target,
                            "status": "not_found",
                            "message": f"{target} process not found"
                        }
                else:
                    return {
                        "type": "error",
                        "message": f"Invalid command value: {command}. Expected 'true' or 'false'"
                    }
            else:
                return {
                    "type": "error",
                    "message": f"Unknown target: {target}"
                }
                
        except subprocess.SubprocessError as e:
            logger.error(f"执行启动命令时出错: {e}")
            return {
                "type": "error",
                "message": f"Subprocess error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"执行启动命令时出错: {e}")
            return {
                "type": "error",
                "message": f"Command execution error: {str(e)}"
            }

    def parse_and_execute_command(self, message_json):
        """解析并执行命令"""
        try:
            data = json.loads(message_json)
            
            # 检查是否包含必需的字段
            if "target" not in data or "command" not in data:
                return {
                    "type": "error",
                    "message": "Missing 'target' or 'command' field in message"
                }
            
            target = data["target"]
            command = data["command"]
            
            logger.info(f"收到启动指令 - 目标: {target}, 命令: {command}")
            
            # 执行启动/停止命令
            return self.execute_launch_command(target, command)

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
        
        # 停止所有正在运行的进程
        for target, process in self.processes.items():
            if process.poll() is None:  # 进程仍在运行
                logger.info(f"停止进程 {target} (PID: {process.pid})")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def run_server():
    """运行服务器的便捷函数"""
    server = AGVStartupWebsocketServer(host="0.0.0.0", port=9001)

    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务器...")
        server.stop_server()
    finally:
        logger.info("服务器已关闭")


if __name__ == "__main__":
    print("AGV Startup WebSocket 服务器")
    print("服务器将监听 ws://0.0.0.0:9001")
    print("接收格式: {\"target\":\"arm_left_start\", \"command\":\"true/false\"}")
    print("按 Ctrl+C 停止服务器")

    run_server()