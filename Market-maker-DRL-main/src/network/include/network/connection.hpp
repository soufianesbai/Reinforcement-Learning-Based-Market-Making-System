#pragma once

#include <functional>
#include <memory>
#include <span>

#include "network/socket.hpp"

namespace hft::network {

using MessageHandler = std::function<void(std::span<const uint8_t>)>;

class Connection {
  public:
    explicit Connection(std::unique_ptr<Socket> socket) : socket_(std::move(socket)) {}

    virtual ~Connection() = default;

    // Delete copy, allow move
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
    Connection(Connection&&) noexcept = default;
    Connection& operator=(Connection&&) noexcept = default;

    bool connect(std::string_view host, uint16_t port) {
        return socket_->connect(host, port);
    }

    bool send(std::span<const uint8_t> data) {
        return socket_->send(data.data(), data.size()) == static_cast<ssize_t>(data.size());
    }

    void set_message_handler(MessageHandler handler) {
        message_handler_ = std::move(handler);
    }

    virtual void poll() = 0;

  protected:
    std::unique_ptr<Socket> socket_;
    MessageHandler message_handler_;
};

}  // namespace hft::network
