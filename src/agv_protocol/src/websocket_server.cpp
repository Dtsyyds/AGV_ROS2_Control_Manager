#include "agv_protocol/websocket_server.hpp"
#include <iostream>

namespace agv_protocol {

WebSocketServerWrapper::WebSocketServerWrapper(int port)
    : port_(port)
    , server_(std::make_unique<Server>())
{
    // 初始化 ASIO
    server_->init_asio();

    // 设置日志级别
    server_->clear_access_channels(websocketpp::log::alevel::all);
    server_->set_access_channels(
        websocketpp::log::alevel::connect |
        websocketpp::log::alevel::disconnect
    );

    server_->clear_error_channels(websocketpp::log::elevel::all);
}

WebSocketServerWrapper::~WebSocketServerWrapper()
{
    stop();
}

void WebSocketServerWrapper::start()
{
    if (running_) {
        return;
    }

    running_ = true;

    // 设置消息处理器
    server_->set_message_handler([this](ConnectionHdl hdl, MessagePtr msg) {
        (void)hdl;  // 避免未使用参数警告
        try {
            std::string payload = msg->get_payload();
            if (message_handler_) {
                message_handler_(payload);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "[WSServer] 处理本地消息异常：" << e.what() << std::endl;
        }
    });

    // 设置连接处理器
    server_->set_open_handler([this](ConnectionHdl hdl) {
        Server::connection_ptr con = server_->get_con_from_hdl(hdl);
        std::cout << "[WSServer] 本地客户端连接：" << con->get_remote_endpoint() << std::endl;
    });

    server_->set_close_handler([](ConnectionHdl hdl) {
        (void)hdl;
        std::cout << "[WSServer] 本地客户端断开连接" << std::endl;
    });

    thread_ = std::make_unique<std::thread>(&WebSocketServerWrapper::runServer, this);
}

void WebSocketServerWrapper::stop()
{
    if (!running_) {
        return;
    }

    running_ = false;

    // 停止服务器
    if (server_) {
        try {
            websocketpp::lib::error_code ec;
            server_->stop_listening(ec);
            server_->stop();  // stop() 不接受参数
        }
        catch (const std::exception& e) {
            std::cerr << "[WSServer] 停止服务器异常：" << e.what() << std::endl;
        }
    }

    // 等待线程结束
    if (thread_ && thread_->joinable()) {
        try {
            thread_->join();
            thread_.reset();
        }
        catch (const std::exception& e) {
            std::cerr << "[WSServer] 等待线程结束时出错：" << e.what() << std::endl;
        }
    }
}

void WebSocketServerWrapper::setMessageHandler(MessageHandler handler)
{
    message_handler_ = std::move(handler);
}

void WebSocketServerWrapper::runServer()
{
    try {
        // 监听端口
        websocketpp::lib::error_code ec;
        server_->listen(port_, ec);
        if (ec) {
            std::cerr << "[WSServer] 监听端口 " << port_ << " 失败：" << ec.message() << std::endl;
            return;
        }

        server_->start_accept(ec);
        if (ec) {
            std::cerr << "[WSServer] 开始接受连接失败：" << ec.message() << std::endl;
            return;
        }

        std::cout << "[WSServer] 本地 WebSocket 服务器已启动，监听端口：" << port_ << std::endl;

        // 运行服务器
        server_->run();
    }
    catch (const std::exception& e) {
        std::cerr << "[WSServer] 本地 WebSocket 服务器异常：" << e.what() << std::endl;
    }
}

} // namespace agv_protocol
