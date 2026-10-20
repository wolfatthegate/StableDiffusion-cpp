// ============================================================================
// http.cpp - Implementation of the minimal HTTP server in http.h.
// ============================================================================
#include "http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace sd {
namespace http {

namespace {

constexpr size_t kMaxHeaderBytes = 64 * 1024;
constexpr size_t kMaxBodyBytes = 4 * 1024 * 1024;

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// Sets SO_NOSIGPIPE on macOS so writes to a closed socket return EPIPE
// instead of killing the process (Linux uses MSG_NOSIGNAL at send time).
void disable_sigpipe(int fd) {
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#else
    (void)fd;
#endif
}

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

}  // namespace

// ---------------------------------------------------------------------------
std::string status_text(int status) {
    switch (status) {
        case 200: return "OK";
        case 201: return "Created";
        case 202: return "Accepted";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 413: return "Payload Too Large";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Unknown";
    }
}

std::string url_decode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit((unsigned char)s[i + 1]) &&
            std::isxdigit((unsigned char)s[i + 2])) {
            out += (char)std::strtol(s.substr(i + 1, 2).c_str(), nullptr, 16);
            i += 2;
        } else if (s[i] == '+') {
            out += ' ';
        } else {
            out += s[i];
        }
    }
    return out;
}

std::string content_type_for(const std::string& path) {
    auto ends = [&](const char* ext) {
        size_t n = std::strlen(ext);
        return path.size() >= n && path.compare(path.size() - n, n, ext) == 0;
    };
    if (ends(".html")) return "text/html; charset=utf-8";
    if (ends(".css")) return "text/css; charset=utf-8";
    if (ends(".js")) return "application/javascript; charset=utf-8";
    if (ends(".json")) return "application/json";
    if (ends(".png")) return "image/png";
    if (ends(".jpg") || ends(".jpeg")) return "image/jpeg";
    if (ends(".svg")) return "image/svg+xml";
    if (ends(".ico")) return "image/x-icon";
    if (ends(".txt")) return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

// ---------------------------------------------------------------------------
std::string Request::header(const std::string& name) const {
    auto it = headers.find(lower(name));
    return it == headers.end() ? "" : it->second;
}

std::string Request::query_param(const std::string& name) const {
    size_t pos = 0;
    while (pos <= query.size()) {
        size_t amp = query.find('&', pos);
        if (amp == std::string::npos) amp = query.size();
        std::string kv = query.substr(pos, amp - pos);
        size_t eq = kv.find('=');
        std::string k = url_decode(kv.substr(0, eq));
        if (k == name) return eq == std::string::npos ? "" : url_decode(kv.substr(eq + 1));
        pos = amp + 1;
    }
    return "";
}

// ---------------------------------------------------------------------------
Connection::~Connection() {
    if (fd_ >= 0) {
        shutdown(fd_, SHUT_RDWR);
        close(fd_);
    }
}

bool Connection::read_request(Request& req) {
    std::string buf;
    char tmp[4096];
    size_t header_end = std::string::npos;
    // Read until the blank line that ends the headers.
    while (header_end == std::string::npos) {
        ssize_t n = recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        buf.append(tmp, (size_t)n);
        if (buf.size() > kMaxHeaderBytes) return false;
        header_end = buf.find("\r\n\r\n");
    }
    // Request line.
    size_t line_end = buf.find("\r\n");
    std::string line = buf.substr(0, line_end);
    size_t sp1 = line.find(' '), sp2 = line.rfind(' ');
    if (sp1 == std::string::npos || sp2 == sp1) return false;
    req.method = line.substr(0, sp1);
    std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = target.find('?');
    req.path = url_decode(target.substr(0, q));
    req.query = q == std::string::npos ? "" : target.substr(q + 1);
    // Headers.
    size_t pos = line_end + 2;
    while (pos < header_end) {
        size_t e = buf.find("\r\n", pos);
        std::string h = buf.substr(pos, e - pos);
        size_t colon = h.find(':');
        if (colon != std::string::npos) {
            std::string k = lower(h.substr(0, colon));
            size_t v = colon + 1;
            while (v < h.size() && h[v] == ' ') ++v;
            req.headers[k] = h.substr(v);
        }
        pos = e + 2;
    }
    // Body.
    size_t content_length = 0;
    std::string cl = req.header("content-length");
    if (!cl.empty()) content_length = (size_t)std::strtoull(cl.c_str(), nullptr, 10);
    if (content_length > kMaxBodyBytes) return false;
    req.body = buf.substr(header_end + 4);
    while (req.body.size() < content_length) {
        ssize_t n = recv(fd_, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        req.body.append(tmp, (size_t)n);
    }
    req.body.resize(content_length);
    return true;
}

bool Connection::write_all(const char* p, size_t n) {
    while (n > 0) {
        ssize_t w = ::send(fd_, p, n, kSendFlags);
        if (w <= 0) return false;
        p += w;
        n -= (size_t)w;
    }
    return true;
}

bool Connection::write(const std::string& data) { return write_all(data.data(), data.size()); }

bool Connection::send(int status, const std::string& content_type, const std::string& body,
                      const std::vector<std::pair<std::string, std::string>>& extra_headers) {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    head += "Content-Type: " + content_type + "\r\n";
    head += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    head += "Connection: close\r\n";
    for (const auto& kv : extra_headers) head += kv.first + ": " + kv.second + "\r\n";
    head += "\r\n";
    return write(head) && write(body);
}

bool Connection::begin_stream(int status, const std::string& content_type) {
    std::string head = "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) + "\r\n";
    head += "Content-Type: " + content_type + "\r\n";
    head += "Cache-Control: no-cache\r\n";
    head += "Connection: close\r\n";
    head += "X-Accel-Buffering: no\r\n\r\n";
    return write(head);
}

// ---------------------------------------------------------------------------
Server::Server(const std::string& host, int port) : port_(port) {
    signal(SIGPIPE, SIG_IGN);
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
        throw std::runtime_error("invalid bind address: " + host);
    if (bind(listen_fd_, (sockaddr*)&addr, sizeof addr) != 0)
        throw std::runtime_error("cannot bind " + host + ":" + std::to_string(port) + " (" + std::strerror(errno) + ")");
    if (listen(listen_fd_, 64) != 0) throw std::runtime_error("listen() failed");
    if (port == 0) {
        socklen_t len = sizeof addr;
        getsockname(listen_fd_, (sockaddr*)&addr, &len);
        port_ = ntohs(addr.sin_port);
    }
}

Server::~Server() {
    if (listen_fd_ >= 0) close(listen_fd_);
}

void Server::route(const std::string& method, const std::string& pattern, Handler handler) {
    routes_.push_back({method, pattern, std::move(handler)});
}

void Server::run() {
    for (;;) {
        sockaddr_in peer{};
        socklen_t len = sizeof peer;
        int fd = accept(listen_fd_, (sockaddr*)&peer, &len);
        if (fd < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("accept() failed");
        }
        disable_sigpipe(fd);
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::thread([this, fd] { handle(fd); }).detach();
    }
}

void Server::handle(int fd) {
    Connection conn(fd);
    Request req;
    if (!conn.read_request(req)) {
        conn.send_text(400, "bad request");
        return;
    }
    bool path_matched = false;
    for (const Route& r : routes_) {
        bool match = r.pattern.size() && r.pattern.back() == '*'
                         ? req.path.compare(0, r.pattern.size() - 1, r.pattern, 0, r.pattern.size() - 1) == 0
                         : req.path == r.pattern;
        if (!match) continue;
        path_matched = true;
        if (r.method != req.method) continue;
        try {
            r.handler(req, conn);
        } catch (const std::exception& e) {
            conn.send_json(500, std::string("{\"error\":\"") + e.what() + "\"}");
        }
        return;
    }
    conn.send_json(path_matched ? 405 : 404, path_matched ? "{\"error\":\"method not allowed\"}" : "{\"error\":\"not found\"}");
}

}  // namespace http
}  // namespace sd
