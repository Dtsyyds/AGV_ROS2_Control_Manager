#!/usr/bin/env python3
"""
测试WebSocket服务器的客户端脚本
"""

import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:9001"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("已连接到WebSocket服务器")
            
            # 发送测试消息
            test_message = {
                "target": "arm_left_start",
                "command": "true"
            }
            
            print(f"发送消息: {json.dumps(test_message)}")
            await websocket.send(json.dumps(test_message))
            
            # 接收响应
            response = await websocket.recv()
            print(f"收到响应: {response}")
            
            # 等待几秒钟再发送停止命令
            await asyncio.sleep(3)
            
            # 发送停止命令
            stop_message = {
                "target": "arm_left_start",
                "command": "false"
            }
            
            print(f"发送停止消息: {json.dumps(stop_message)}")
            await websocket.send(json.dumps(stop_message))
            
            # 接收响应
            response = await websocket.recv()
            print(f"收到响应: {response}")
            
    except Exception as e:
        print(f"连接WebSocket服务器时出错: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())