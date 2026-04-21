using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;
using System.Net.WebSockets;
using System.Text;
using System.IO;
using Newtonsoft.Json;
using System.Runtime.Serialization;

namespace AGVWebSocketServer
{
    public class AGVWebSocketServer
    {
        private readonly string host;
        private readonly int port;
        private List<WebSocket> clients;
        private bool isRunning;
        private CancellationTokenSource cancellationTokenSource;
        
        // 机械臂关节位置（模拟）
        private double[] jointPositions = new double[6] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        
        public AGVWebSocketServer(string host = "0.0.0.0", int port = 9000)
        {
            this.host = host;
            this.port = port;
            this.clients = new List<WebSocket>();
            this.isRunning = false;
            this.cancellationTokenSource = new CancellationTokenSource();
        }
        
        public async Task StartAsync()
        {
            isRunning = true;
            var webSocketServer = new System.Net.HttpListener();
            webSocketServer.Prefixes.Add($"http://{host}:{port}/");
            webSocketServer.Start();
            
            Console.WriteLine($"WebSocket服务器启动在 {host}:{port}");
            
            while (isRunning)
            {
                try
                {
                    var context = await webSocketServer.GetContextAsync();
                    
                    if (context.Request.IsWebSocketRequest)
                    {
                        var webSocketContext = await context.AcceptWebSocketAsync(null);
                        var webSocket = webSocketContext.WebSocket;
                        
                        Console.WriteLine($"新客户端连接: {context.Request.RemoteEndPoint}");
                        
                        // 注册客户端
                        lock (clients)
                        {
                            clients.Add(webSocket);
                        }
                        
                        // 发送连接确认
                        var connectionMsg = new
                        {
                            type = "connection_ack",
                            server = "agv_control_server",
                            timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString()
                        };
                        await SendToClientAsync(webSocket, connectionMsg);
                        
                        // 处理客户端消息
                        _ = HandleClientAsync(webSocket, context.Request.RemoteEndPoint.ToString());
                    }
                    else
                    {
                        context.Response.StatusCode = 400;
                        context.Response.Close();
                    }
                }
                catch (Exception ex)
                {
                    if (isRunning)
                    {
                        Console.WriteLine($"接受客户端连接时出错: {ex.Message}");
                    }
                }
            }
            
            webSocketServer.Close();
        }
        
        private async Task HandleClientAsync(WebSocket webSocket, string clientEndpoint)
        {
            try
            {
                var buffer = new byte[1024 * 4];
                
                while (webSocket.State == WebSocketState.Open && isRunning)
                {
                    var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), cancellationTokenSource.Token);
                    
                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        var message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        Console.WriteLine($"收到客户端消息: {message}");
                        
                        var response = ParseAndExecuteCommand(message);
                        if (response != null)
                        {
                            await SendToClientAsync(webSocket, response);
                        }
                        
                        // 广播状态更新
                        await BroadcastStatusUpdateAsync();
                    }
                    else if (result.MessageType == WebSocketMessageType.Close)
                    {
                        Console.WriteLine($"客户端请求关闭连接: {clientEndpoint}");
                        await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"处理客户端消息时出错: {ex.Message}");
            }
            finally
            {
                // 从客户端列表中移除
                lock (clients)
                {
                    clients.Remove(webSocket);
                }
                
                if (webSocket.State != WebSocketState.Closed && webSocket.State != WebSocketState.Aborted)
                {
                    try
                    {
                        await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, "Error occurred", CancellationToken.None);
                    }
                    catch
                    {
                        // 忽略关闭错误
                    }
                }
                
                Console.WriteLine($"客户端断开: {clientEndpoint}");
            }
        }
        
        private async Task SendToClientAsync(WebSocket webSocket, object message)
        {
            try
            {
                var jsonString = JsonConvert.SerializeObject(message);
                var bytes = Encoding.UTF8.GetBytes(jsonString);
                
                await webSocket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"发送消息失败: {ex.Message}");
            }
        }
        
        private async Task BroadcastStatusUpdateAsync()
        {
            var statusMsg = new
            {
                type = "current_positions",
                data = new
                {
                    joint1 = jointPositions[0],
                    joint2 = jointPositions[1],
                    joint3 = jointPositions[2],
                    joint4 = jointPositions[3],
                    joint5 = jointPositions[4],
                    joint6 = jointPositions[5]
                },
                timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString()
            };
            
            var message = JsonConvert.SerializeObject(statusMsg);
            var bytes = Encoding.UTF8.GetBytes(message);
            
            List<WebSocket> clientsCopy;
            lock (clients)
            {
                clientsCopy = new List<WebSocket>(clients);
            }
            
            foreach (var client in clientsCopy)
            {
                try
                {
                    if (client.State == WebSocketState.Open)
                    {
                        await client.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, cancellationTokenSource.Token);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"广播消息失败: {ex.Message}");
                    lock (clients)
                    {
                        clients.Remove(client);
                    }
                }
            }
        }
        
        private object ParseAndExecuteCommand(string messageJson)
        {
            try
            {
                dynamic data = JsonConvert.DeserializeObject(messageJson);
                string commandType = data.type;
                
                switch (commandType)
                {
                    case "connection":
                        Console.WriteLine($"收到连接确认: {messageJson}");
                        return new { status = "connected", message = "Server connected successfully" };
                    
                    case "joint_control":
                        var jointData = data.data;
                        var joints = new double[6];
                        joints[0] = jointData?.joint1 ?? jointPositions[0];
                        joints[1] = jointData?.joint2 ?? jointPositions[1];
                        joints[2] = jointData?.joint3 ?? jointPositions[2];
                        joints[3] = jointData?.joint4 ?? jointPositions[3];
                        joints[4] = jointData?.joint5 ?? jointPositions[4];
                        joints[5] = jointData?.joint6 ?? jointPositions[5];
                        
                        jointPositions = joints;
                        Console.WriteLine($"执行关节控制命令: [{string.Join(", ", joints)}]");
                        
                        return new
                        {
                            type = "command_ack",
                            command = "joint_control",
                            status = "executed",
                            positions = joints
                        };
                    
                    case "move_to_position":
                        var positionData = data.data;
                        var posJoints = new double[6];
                        posJoints[0] = positionData?.j1 ?? jointPositions[0];
                        posJoints[1] = positionData?.j2 ?? jointPositions[1];
                        posJoints[2] = positionData?.j3 ?? jointPositions[2];
                        posJoints[3] = positionData?.j4 ?? jointPositions[3];
                        posJoints[4] = positionData?.j5 ?? jointPositions[4];
                        posJoints[5] = positionData?.j6 ?? jointPositions[5];
                        
                        jointPositions = posJoints;
                        Console.WriteLine($"执行移动到位置命令: [{string.Join(", ", posJoints)}]");
                        
                        return new
                        {
                            type = "command_ack",
                            command = "move_to_position",
                            status = "executed",
                            positions = posJoints
                        };
                    
                    case "emergency_stop":
                        Console.WriteLine("执行紧急停止命令");
                        // 紧急停止逻辑可以在这里实现
                        return new
                        {
                            type = "command_ack",
                            command = "emergency_stop",
                            status = "executed"
                        };
                    
                    case "power_on":
                        Console.WriteLine("执行上电命令");
                        // 上电逻辑可以在这里实现
                        return new
                        {
                            type = "command_ack",
                            command = "power_on",
                            status = "executed"
                        };
                    
                    case "power_off":
                        Console.WriteLine("执行断电命令");
                        // 断电逻辑可以在这里实现
                        return new
                        {
                            type = "command_ack",
                            command = "power_off",
                            status = "executed"
                        };
                    
                    case "get_status":
                        // 返回当前状态
                        return new
                        {
                            type = "status_response",
                            joint_positions = jointPositions,
                            timestamp = DateTimeOffset.Now.ToUnixTimeMilliseconds().ToString()
                        };
                    
                    default:
                        Console.WriteLine($"未知命令类型: {commandType}");
                        return new
                        {
                            type = "error",
                            message = $"Unknown command type: {commandType}"
                        };
                }
            }
            catch (JsonSerializationException ex)
            {
                Console.WriteLine($"JSON解析错误: {ex.Message}");
                return new
                {
                    type = "error",
                    message = $"Invalid JSON: {ex.Message}"
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"执行命令时出错: {ex.Message}");
                return new
                {
                    type = "error",
                    message = $"Command execution error: {ex.Message}"
                };
            }
        }
        
        public void Stop()
        {
            isRunning = false;
            cancellationTokenSource.Cancel();
            Console.WriteLine("正在停止WebSocket服务器...");
        }
        
        public static async Task Main(string[] args)
        {
            Console.WriteLine("AGV WebSocket 服务器");
            Console.WriteLine("服务器将监听 ws://0.0.0.0:9000");
            Console.WriteLine("按任意键停止服务器...");
            
            var server = new AGVWebSocketServer("0.0.0.0", 9000);
            
            // 在后台任务中启动服务器
            var serverTask = server.StartAsync();
            
            Console.ReadKey();
            server.Stop();
            
            await serverTask;
            
            Console.WriteLine("服务器已停止");
        }
    }
}