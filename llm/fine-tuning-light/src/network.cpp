#include "network.hpp"
#include <cstring>
#include <iostream>
#include <chrono>
#include <thread>

namespace net {

// ---------- Socket ----------
Socket::Socket(int fd_) : fd(fd_) {}
Socket::~Socket() { close(); }
Socket::Socket(Socket&& other) noexcept : fd(other.fd) { other.fd = -1; }
Socket& Socket::operator=(Socket&& other) noexcept {
    if (this != &other) {
        close();
        fd = other.fd;
        other.fd = -1;
    }
    return *this;
}

void Socket::close() {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}

void Socket::set_nonblocking(bool on) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (on) flags |= O_NONBLOCK;
    else flags &= ~O_NONBLOCK;
    fcntl(fd, F_SETFL, flags);
}

void Socket::set_tcp_nodelay(bool on) {
    int val = on ? 1 : 0;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &val, sizeof(val));
}

void Socket::set_reuse_addr(bool on) {
    int val = on ? 1 : 0;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
}

void Socket::set_recv_buffer(int size) {
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size));
}
void Socket::set_send_buffer(int size) {
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size));
}

bool Socket::send_all(const uint8_t* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::send(fd, data + sent, len - sent, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            return false;
        }
        if (n == 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

bool Socket::recv_all(uint8_t* data, size_t len) {
    size_t received = 0;
    while (received < len) {
        ssize_t n = ::recv(fd, data + received, len - received, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }
            return false;
        }
        if (n == 0) return false; // peer closed
        received += static_cast<size_t>(n);
    }
    return true;
}

bool Socket::send_message(const std::vector<uint8_t>& msg) {
    uint32_t len_be = htonl(static_cast<uint32_t>(msg.size()));
    if (!send_all(reinterpret_cast<const uint8_t*>(&len_be), sizeof(len_be))) return false;
    if (!msg.empty()) return send_all(msg.data(), msg.size());
    return true;
}

bool Socket::recv_message(std::vector<uint8_t>& msg) {
    uint32_t len_be = 0;
    if (!recv_all(reinterpret_cast<uint8_t*>(&len_be), sizeof(len_be))) return false;
    uint32_t len = ntohl(len_be);
    msg.resize(len);
    if (len > 0 && !recv_all(msg.data(), len)) return false;
    return true;
}

bool Socket::send_floats(const std::vector<float>& data) {
    size_t nbytes = data.size() * sizeof(float);
    std::vector<uint8_t> buf(nbytes);
    std::memcpy(buf.data(), data.data(), nbytes);
    return send_message(buf);
}

bool Socket::recv_floats(std::vector<float>& data, size_t num_floats) {
    std::vector<uint8_t> buf;
    if (!recv_message(buf)) return false;
    size_t expected = num_floats * sizeof(float);
    if (buf.size() != expected) return false;
    data.resize(num_floats);
    std::memcpy(data.data(), buf.data(), expected);
    return true;
}

// ---------- ServerSocket ----------
bool ServerSocket::bind_and_listen(const std::string& host, int port, int backlog) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sock = Socket(fd);
    sock.set_reuse_addr(true);
    sock.set_tcp_nodelay(true);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (host.empty() || host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) return false;
    }

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind failed: " << strerror(errno) << "\n";
        return false;
    }
    if (::listen(fd, backlog) < 0) return false;
    return true;
}

Socket ServerSocket::accept_connection() {
    sockaddr_in client{};
    socklen_t len = sizeof(client);
    int cfd = ::accept(sock.fd, reinterpret_cast<sockaddr*>(&client), &len);
    if (cfd < 0) return Socket(-1);
    Socket client_sock(cfd);
    client_sock.set_tcp_nodelay(true);
    return client_sock;
}

// ---------- Client ----------
Socket connect_to(const std::string& host, int port, int timeout_ms) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return Socket(-1);

    // Resolve hostname
    hostent* he = gethostbyname(host.c_str());
    if (!he) {
        close(fd);
        return Socket(-1);
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    std::memcpy(&addr.sin_addr, he->h_addr_list[0], static_cast<size_t>(he->h_length));

    // Set non-blocking for timeout-capable connect
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

    int rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (rc < 0 && errno == EINPROGRESS) {
        pollfd pfd{fd, POLLOUT, 0};
        rc = poll(&pfd, 1, timeout_ms);
        if (rc <= 0) {
            close(fd);
            return Socket(-1);
        }
        int so_error = 0;
        socklen_t len = sizeof(so_error);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len);
        if (so_error != 0) {
            close(fd);
            return Socket(-1);
        }
    } else if (rc < 0) {
        close(fd);
        return Socket(-1);
    }

    // Restore blocking mode
    fcntl(fd, F_SETFL, flags);
    Socket s(fd);
    s.set_tcp_nodelay(true);
    return s;
}

std::string get_hostname() {
    char buf[256];
    if (gethostname(buf, sizeof(buf)) == 0) return std::string(buf);
    return "localhost";
}

std::string ip_from_fd(int fd) {
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    if (getpeername(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &addr.sin_addr, ip, sizeof(ip));
        return std::string(ip);
    }
    return "unknown";
}

} // namespace net
