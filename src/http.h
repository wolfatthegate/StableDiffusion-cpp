// ============================================================================
// http.h - A small HTTP/1.1 server on POSIX sockets, enough for a local web
// UI: routing, JSON bodies, static files and streamed (Server-Sent Events)
// responses. One thread per connection; every response closes the
// connection, so there is no keep-alive bookkeeping.
//
//     http::Server server("127.0.0.1", 8080);
//     server.route("GET", "/api/status", [](const http::Request& req, http::Connection& conn) {
//         conn.send_json(200, "{\"ok\":true}");
//     });
//     server.route("GET", "/api/jobs/*", ...);     // trailing * matches any suffix
//     server.run();                                 // blocks
//
// This is for trusted local use. It has request size limits but no TLS,
// authentication, compression or HTTP/2.
// ============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace sd {
namespace http {

struct Request {
    std::string method;                          // "GET", "POST", ...
    std::string path;                            // decoded path without the query string
    std::string query;                           // raw text after '?', or empty
    std::map<std::string, std::string> headers;  // keys lower-cased
    std::string body;

    std::string header(const std::string& name) const;  // "" if absent (name any case)
    std::string query_param(const std::string& name) const;  // percent-decoded, "" if absent
};

// One client connection. Handlers either call send_* once, or begin_stream()
// followed by any number of write() calls for a streamed response.
class Connection {
public:
    explicit Connection(int fd) : fd_(fd) {}
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // Reads and parses one request. Returns false on a malformed request or
    // a closed socket.
    bool read_request(Request& req);

    bool send(int status, const std::string& content_type, const std::string& body,
              const std::vector<std::pair<std::string, std::string>>& extra_headers = {});
    bool send_json(int status, const std::string& json) { return send(status, "application/json", json); }
    bool send_text(int status, const std::string& text) { return send(status, "text/plain; charset=utf-8", text); }
    // Status line and headers for a response of unknown length; follow with write().
    bool begin_stream(int status, const std::string& content_type);
    // Writes raw bytes; returns false once the client has gone away.
    bool write(const std::string& data);

private:
    int fd_;
    bool write_all(const char* p, size_t n);
};

using Handler = std::function<void(const Request&, Connection&)>;

class Server {
public:
    // Binds and listens; throws std::runtime_error if the port is taken.
    Server(const std::string& host, int port);
    ~Server();
    // Registers a handler. `pattern` is an exact path, or a prefix ending in
    // '*'. The first registered match wins; a path match with the wrong
    // method yields 405, no match 404.
    void route(const std::string& method, const std::string& pattern, Handler handler);
    // Accepts connections forever, one detached thread each.
    void run();
    int port() const { return port_; }

private:
    struct Route {
        std::string method, pattern;
        Handler handler;
    };
    int listen_fd_ = -1;
    int port_;
    std::vector<Route> routes_;
    void handle(int fd);
};

// Helpers for building responses by hand.
std::string status_text(int status);
std::string url_decode(const std::string& s);
// Content type from a file extension (".html", ".png", ...), default octet-stream.
std::string content_type_for(const std::string& path);

}  // namespace http
}  // namespace sd
