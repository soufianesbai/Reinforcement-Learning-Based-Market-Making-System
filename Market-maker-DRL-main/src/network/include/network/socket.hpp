#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace hft::network {

enum class SocketType { TCP, UDP };

enum class SocketOption {
    REUSEADDR,
    REUSEPORT,
    NODELAY,
    QUICKACK,
    NONBLOCKING,
    RCVBUF_SIZE,
    SNDBUF_SIZE
};

class Socket {
  public:
    Socket(SocketType type) : type_(type), fd_(-1) {}
    virtual ~Socket() = default;

    // Delete copy, allow move
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&&) noexcept = default;
    Socket& operator=(Socket&&) noexcept = default;

    virtual bool connect(std::string_view host, uint16_t port) = 0;
    virtual bool bind(std::string_view host, uint16_t port) = 0;
    virtual ssize_t send(const void* data, size_t size) = 0;
    virtual ssize_t receive(void* buffer, size_t size) = 0;
    virtual void close() = 0;

    [[nodiscard]] bool is_connected() const {
        return fd_ >= 0;
    }

  protected:
    SocketType type_;
    int fd_;
};

}  // namespace hft::network
