#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <stdexcept>

// Cross-platform POSIX sockets (Linux/macOS)
#ifdef __linux__
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <netdb.h>
    #include <poll.h>
    #include <errno.h>
#elif __APPLE__
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <netdb.h>
    #include <poll.h>
    #include <errno.h>
#endif

namespace net {

class Socket {
public:
    int fd = -1;
    Socket() = default;
    explicit Socket(int fd_);
    ~Socket();
    Socket(Socket&& other) noexcept;
    Socket& operator=(Socket&& other) noexcept;
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    void close();
    bool valid() const { return fd >= 0; }
    void set_nonblocking(bool on);
    void set_tcp_nodelay(bool on);
    void set_reuse_addr(bool on);
    void set_recv_buffer(int size);
    void set_send_buffer(int size);

    // Send/receive with length-prefix framing (uint32_t BE)
    bool send_all(const uint8_t* data, size_t len);
    bool recv_all(uint8_t* data, size_t len);
    bool send_message(const std::vector<uint8_t>& msg);
    bool recv_message(std::vector<uint8_t>& msg); // blocking

    bool send_floats(const std::vector<float>& data);
    bool recv_floats(std::vector<float>& data, size_t num_floats);
};

// TCP server socket for accepting connections
class ServerSocket {
public:
    Socket sock;
    bool bind_and_listen(const std::string& host, int port, int backlog = 64);
    Socket accept_connection();
};

// TCP client
Socket connect_to(const std::string& host, int port, int timeout_ms = 10000);

// Address utilities
std::string get_hostname();
std::string ip_from_fd(int fd);

} // namespace net
