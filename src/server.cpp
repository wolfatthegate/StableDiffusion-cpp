// ============================================================================
// server.cpp - `sd-server`: serves the web UI in web/ and a JSON API on top of
// Pipeline. Generations run one at a time on a worker thread; clients follow
// progress over Server-Sent Events and receive a live preview every step.
//
// API
//   GET  /                         the web page (web/index.html)
//   GET  /api/status               {"model_dir","threads","queue","busy","current"}
//   GET  /api/jobs                 finished jobs, newest first (for the gallery)
//   POST /api/generate             body: GenerateParams as JSON -> {"id"}
//   GET  /api/jobs/<id>            snapshot of one job
//   GET  /api/jobs/<id>/events     SSE stream: queued, started, phase, step, done, error, cancelled
//   POST /api/jobs/<id>/cancel     stops after the current step
//   DELETE /api/jobs/<id>          forgets a finished (or cancelled/failed) job and its image
//   GET  /api/jobs/<id>/image.png  the finished image
// ============================================================================
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#include "http.h"
#include "image.h"
#include "json.h"
#include "pipeline.h"

using namespace sd;

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// One generation request and everything the UI needs to know about it.
struct Job {
    std::string id;
    GenerateParams params;
    double created_ms = now_ms();

    std::mutex m;
    std::condition_variable cv;
    std::vector<std::string> events;  // SSE-formatted, in order; replayed to late subscribers
    std::string status = "queued";    // queued | running | done | error | cancelled
    bool finished = false;
    std::atomic<bool> cancel{false};
    std::vector<uint8_t> png;         // final image once done
    int64_t seed = 0;
    double seconds = 0;
    std::string error;

    void emit(const std::string& event, const std::string& json) {
        std::lock_guard<std::mutex> lk(m);
        events.push_back("event: " + event + "\ndata: " + json + "\n\n");
        cv.notify_all();
    }
    void finish(const std::string& final_status) {
        std::lock_guard<std::mutex> lk(m);
        status = final_status;
        finished = true;
        cv.notify_all();
    }
};

// Serialises generations: one worker thread drains a FIFO of jobs.
class JobQueue {
public:
    explicit JobQueue(const Pipeline& pipe) : pipe_(pipe), worker_([this] { loop(); }) {}

    std::shared_ptr<Job> submit(const GenerateParams& p) {
        auto job = std::make_shared<Job>();
        job->params = p;
        job->id = new_id();
        size_t position;
        {
            std::lock_guard<std::mutex> lk(m_);
            jobs_[job->id] = job;
            order_.push_back(job->id);
            pending_.push_back(job);
            position = pending_.size() - 1 + (busy_ ? 1 : 0);
            trim_locked();
        }
        job->emit("queued", "{\"id\":" + json_quote(job->id) + ",\"position\":" + std::to_string(position) + "}");
        cv_.notify_one();
        return job;
    }

    std::shared_ptr<Job> find(const std::string& id) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = jobs_.find(id);
        return it == jobs_.end() ? nullptr : it->second;
    }

    // Finished jobs, newest first.
    std::vector<std::shared_ptr<Job>> finished() {
        std::lock_guard<std::mutex> lk(m_);
        std::vector<std::shared_ptr<Job>> out;
        for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
            auto job = jobs_[*it];
            std::lock_guard<std::mutex> jl(job->m);
            if (job->status == "done") out.push_back(job);
        }
        return out;
    }

    size_t pending() {
        std::lock_guard<std::mutex> lk(m_);
        return pending_.size();
    }
    bool busy() {
        std::lock_guard<std::mutex> lk(m_);
        return busy_;
    }
    // Id of the job being generated right now, or "" when idle.
    std::string current() {
        std::lock_guard<std::mutex> lk(m_);
        return busy_ ? current_ : "";
    }
    // Drops a finished job and frees its image. Returns false if the job is
    // unknown, or still queued or running (cancel it first).
    bool remove(const std::string& id) {
        std::lock_guard<std::mutex> lk(m_);
        auto it = jobs_.find(id);
        if (it == jobs_.end()) return false;
        {
            std::lock_guard<std::mutex> jl(it->second->m);
            if (!it->second->finished) return false;
        }
        jobs_.erase(it);
        order_.erase(std::remove(order_.begin(), order_.end(), id), order_.end());
        return true;
    }

private:
    const Pipeline& pipe_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> pending_;
    std::map<std::string, std::shared_ptr<Job>> jobs_;
    std::deque<std::string> order_;
    bool busy_ = false;
    std::string current_;
    std::thread worker_;
    static constexpr size_t kKeepJobs = 60;

    std::string new_id() {
        static std::mt19937_64 rng{std::random_device{}()};
        static std::mutex rm;
        std::lock_guard<std::mutex> lk(rm);
        char buf[20];
        std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)rng());
        return buf;
    }

    // Drops the oldest finished jobs so memory stays bounded.
    void trim_locked() {
        while (order_.size() > kKeepJobs) {
            auto job = jobs_[order_.front()];
            bool done;
            {
                std::lock_guard<std::mutex> jl(job->m);
                done = job->finished;
            }
            if (!done) break;
            jobs_.erase(order_.front());
            order_.pop_front();
        }
    }

    void loop() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return !pending_.empty(); });
                job = pending_.front();
                pending_.pop_front();
                busy_ = true;
                current_ = job->id;
            }
            run(*job);
            std::lock_guard<std::mutex> lk(m_);
            busy_ = false;
        }
    }

    void run(Job& job) {
        if (job.cancel) {
            job.emit("cancelled", "{}");
            job.finish("cancelled");
            return;
        }
        {
            std::lock_guard<std::mutex> lk(job.m);
            job.status = "running";
        }
        job.emit("started", "{}");
        const double t0 = now_ms();
        double step_total_ms = 0;
        int steps_done = 0;
        std::string last_phase;
        try {
            GenerateResult r = pipe_.generate(job.params, [&](const Progress& pr) {
                if (job.cancel) return false;
                const char* phase = pr.phase == Progress::TextEncoding ? "text"
                                    : pr.phase == Progress::Denoising ? "denoise"
                                                                      : "decode";
                if (phase != last_phase) {
                    last_phase = phase;
                    job.emit("phase", std::string("{\"phase\":\"") + phase + "\"}");
                }
                if (pr.phase != Progress::Denoising) return true;
                step_total_ms += pr.step_seconds * 1000.0;
                steps_done = pr.step;
                const double avg = step_total_ms / steps_done;
                // Decoding costs roughly 1.7 SD steps, or 0.3 Z-Image steps.
                const double decode_steps = pipe_.family() == Pipeline::Family::ZImage ? 0.3 : 1.7;
                const double eta = avg * (pr.total - pr.step) + decode_steps * avg;
                std::ostringstream js;
                js << "{\"step\":" << pr.step << ",\"total\":" << pr.total << ",\"timestep\":" << pr.timestep
                   << ",\"step_ms\":" << (long)(pr.step_seconds * 1000.0) << ",\"elapsed_ms\":" << (long)(now_ms() - t0)
                   << ",\"eta_ms\":" << (long)eta;
                if (pr.preview) {
                    int w, h;
                    std::vector<uint8_t> rgb;
                    Pipeline::latent_to_rgb(*pr.preview, w, h, rgb);
                    std::vector<uint8_t> png = encode_png(w, h, rgb);
                    js << ",\"preview\":\"data:image/png;base64," << base64_encode(png.data(), png.size()) << "\""
                       << ",\"preview_w\":" << w << ",\"preview_h\":" << h;
                }
                js << "}";
                job.emit("step", js.str());
                return true;
            });
            if (r.cancelled) {
                job.emit("cancelled", "{}");
                job.finish("cancelled");
                return;
            }
            {
                std::lock_guard<std::mutex> lk(job.m);
                job.png = encode_png(r.width, r.height, r.rgb);
                job.seed = r.seed;
                job.seconds = r.seconds;
            }
            std::ostringstream js;
            js << "{\"image\":\"/api/jobs/" << job.id << "/image.png\",\"seed\":" << r.seed << ",\"seconds\":" << r.seconds
               << ",\"width\":" << r.width << ",\"height\":" << r.height << "}";
            job.emit("done", js.str());
            job.finish("done");
        } catch (const std::exception& e) {
            {
                std::lock_guard<std::mutex> lk(job.m);
                job.error = e.what();
            }
            job.emit("error", "{\"message\":" + json_quote(e.what()) + "}");
            job.finish("error");
        }
    }
};

// Reads GenerateParams from a JSON object; missing fields keep the defaults.
GenerateParams params_from_json(const std::string& body, const GenerateParams& defaults) {
    Json j = parse_json(body);
    if (j.type != Json::Object) throw std::runtime_error("expected a JSON object");
    GenerateParams p = defaults;
    auto str = [&](const char* k, std::string& out) {
        if (const Json* v = j.find(k)) {
            if (v->type != Json::String) throw std::runtime_error(std::string(k) + " must be a string");
            out = v->str;
        }
    };
    auto num = [&](const char* k, double& out) {
        if (const Json* v = j.find(k)) {
            if (v->type != Json::Number) throw std::runtime_error(std::string(k) + " must be a number");
            out = v->num;
        }
    };
    str("prompt", p.prompt);
    str("negative", p.negative);
    str("sampler", p.sampler);
    double steps = p.steps, cfg = p.cfg, width = p.width, height = p.height, seed = (double)p.seed;
    num("steps", steps);
    num("cfg", cfg);
    num("width", width);
    num("height", height);
    num("seed", seed);
    p.steps = (int)steps;
    p.cfg = (float)cfg;
    p.width = (int)width;
    p.height = (int)height;
    p.seed = (int64_t)seed;
    return p;
}

std::string job_json(Job& job) {
    std::lock_guard<std::mutex> lk(job.m);
    std::ostringstream js;
    js << "{\"id\":" << json_quote(job.id) << ",\"status\":" << json_quote(job.status)
       << ",\"prompt\":" << json_quote(job.params.prompt) << ",\"negative\":" << json_quote(job.params.negative)
       << ",\"sampler\":" << json_quote(job.params.sampler) << ",\"steps\":" << job.params.steps
       << ",\"cfg\":" << job.params.cfg << ",\"width\":" << job.params.width << ",\"height\":" << job.params.height
       << ",\"seed\":" << job.seed << ",\"seconds\":" << job.seconds << ",\"created_ms\":" << (long long)job.created_ms;
    if (job.status == "done") js << ",\"image\":\"/api/jobs/" << job.id << "/image.png\"";
    if (!job.error.empty()) js << ",\"error\":" << json_quote(job.error);
    js << "}";
    return js.str();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Options {
    std::string model_dir = "models/sd15";
    std::string host = "127.0.0.1";
    std::string web_dir = "web";
    int port = 8080;
    int threads = 0;
};

void usage() {
    std::fprintf(stderr,
                 "usage: sd-server [options]\n"
                 "  --model-dir DIR   diffusers-layout model directory (default models/sd15)\n"
                 "  --host ADDR       bind address (default 127.0.0.1; 0.0.0.0 for the LAN)\n"
                 "  --port N          TCP port (default 8080)\n"
                 "  --web-dir DIR     directory holding index.html (default web)\n"
                 "  --threads N       worker threads (default: all cores)\n");
}

}  // namespace

int main(int argc, char** argv) try {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model-dir" || a == "-m") opt.model_dir = need();
        else if (a == "--host") opt.host = need();
        else if (a == "--port" || a == "-p") opt.port = std::atoi(need().c_str());
        else if (a == "--web-dir") opt.web_dir = need();
        else if (a == "--threads" || a == "-t") opt.threads = std::atoi(need().c_str());
        else {
            usage();
            std::exit(a == "--help" || a == "-h" ? 0 : 2);
        }
    }
    if (opt.threads > 0) set_num_threads(opt.threads);

    std::fprintf(stderr, "loading %s ...\n", opt.model_dir.c_str());
    Pipeline pipe(opt.model_dir);
    std::fprintf(stderr, "loaded (unet %.0fM params), %d threads\n", pipe.unet_params() / 1e6, num_threads());
    JobQueue queue(pipe);

    http::Server server(opt.host, opt.port);

    server.route("GET", "/", [&](const http::Request&, http::Connection& c) {
        std::string page = read_file(opt.web_dir + "/index.html");
        if (page.empty()) {
            c.send_text(500, "web/index.html not found; run from the repository root or pass --web-dir");
            return;
        }
        c.send(200, "text/html; charset=utf-8", page, {{"Cache-Control", "no-cache"}});
    });

    server.route("GET", "/api/status", [&](const http::Request&, http::Connection& c) {
        const GenerateParams d = pipe.defaults();
        std::ostringstream js;
        js << "{\"model_dir\":" << json_quote(pipe.model_dir()) << ",\"family\":" << json_quote(pipe.family_name())
           << ",\"threads\":" << num_threads() << ",\"queue\":" << queue.pending()
           << ",\"busy\":" << (queue.busy() ? "true" : "false") << ",\"current\":" << json_quote(queue.current())
           << ",\"defaults\":{\"steps\":" << d.steps << ",\"cfg\":" << d.cfg << ",\"width\":" << d.width
           << ",\"height\":" << d.height << ",\"sampler\":" << json_quote(d.sampler) << "},\"samplers\":[";
        bool first = true;
        for (const auto& s : pipe.samplers()) {
            js << (first ? "" : ",") << json_quote(s);
            first = false;
        }
        js << "]}";
        c.send_json(200, js.str());
    });

    server.route("GET", "/api/jobs", [&](const http::Request&, http::Connection& c) {
        std::string out = "[";
        bool first = true;
        for (auto& job : queue.finished()) {
            out += (first ? "" : ",") + job_json(*job);
            first = false;
        }
        c.send_json(200, out + "]");
    });

    server.route("POST", "/api/generate", [&](const http::Request& req, http::Connection& c) {
        GenerateParams p;
        try {
            p = params_from_json(req.body, pipe.defaults());
            if (p.prompt.empty()) throw std::runtime_error("prompt is empty");
            if (p.steps < 1 || p.steps > 500) throw std::runtime_error("steps must be between 1 and 500");
            if (p.width % 64 || p.height % 64 || p.width < 64 || p.height < 64 || p.width > 2048 || p.height > 2048)
                throw std::runtime_error("width and height must be multiples of 64 between 64 and 2048");
            const auto samplers = pipe.samplers();
            if (std::find(samplers.begin(), samplers.end(), p.sampler) == samplers.end())
                throw std::runtime_error("sampler not available for this model");
        } catch (const std::exception& e) {
            c.send_json(400, "{\"error\":" + json_quote(e.what()) + "}");
            return;
        }
        auto job = queue.submit(p);
        c.send_json(202, "{\"id\":" + json_quote(job->id) + "}");
    });

    // /api/jobs/<id>[/events|/cancel|/image.png]
    auto job_route = [&](const http::Request& req, http::Connection& c) {
        std::string rest = req.path.substr(std::string("/api/jobs/").size());
        size_t slash = rest.find('/');
        std::string id = rest.substr(0, slash), action = slash == std::string::npos ? "" : rest.substr(slash + 1);
        auto job = queue.find(id);
        if (!job) {
            c.send_json(404, "{\"error\":\"no such job\"}");
            return;
        }
        if (action.empty() && req.method == "GET") {
            c.send_json(200, job_json(*job));
        } else if (action.empty() && req.method == "DELETE") {
            if (queue.remove(id))
                c.send_json(200, "{\"ok\":true}");
            else
                c.send_json(409, "{\"error\":\"job is still running; cancel it first\"}");
        } else if (action == "cancel" && req.method == "POST") {
            job->cancel = true;
            c.send_json(200, "{\"ok\":true}");
        } else if (action == "image.png" && req.method == "GET") {
            std::string png;
            {
                std::lock_guard<std::mutex> lk(job->m);
                png.assign(job->png.begin(), job->png.end());
            }
            if (png.empty()) {
                c.send_json(404, "{\"error\":\"image not ready\"}");
                return;
            }
            c.send(200, "image/png", png, {{"Cache-Control", "max-age=86400"}});
        } else if (action == "events" && req.method == "GET") {
            if (!c.begin_stream(200, "text/event-stream")) return;
            size_t sent = 0;
            for (;;) {
                std::vector<std::string> batch;
                bool finished;
                {
                    std::unique_lock<std::mutex> lk(job->m);
                    job->cv.wait_for(lk, std::chrono::seconds(15),
                                     [&] { return job->events.size() > sent || job->finished; });
                    batch.assign(job->events.begin() + (long)sent, job->events.end());
                    sent = job->events.size();
                    finished = job->finished;
                }
                if (batch.empty() && !finished) {
                    if (!c.write(": keep-alive\n\n")) return;  // comment line keeps proxies happy
                    continue;
                }
                for (const auto& ev : batch)
                    if (!c.write(ev)) return;
                if (finished) return;
            }
        } else {
            c.send_json(404, "{\"error\":\"not found\"}");
        }
    };
    server.route("GET", "/api/jobs/*", job_route);
    server.route("POST", "/api/jobs/*", job_route);
    server.route("DELETE", "/api/jobs/*", job_route);

    std::fprintf(stderr, "listening on http://%s:%d/\n", opt.host.c_str(), server.port());
    server.run();
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
}
