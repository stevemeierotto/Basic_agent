/*
 * Copyright (c) 2026 Steve Meierotto
 *
 * Thoth — synthetic EGAR-LX A/B orchestration
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/synthetic_ab_pilot.h"
#include "../include/alp_sha256.h"
#include "../include/embedding_engine.h"

#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace Thoth {
namespace {

std::string hashFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    in.seekg(0, std::ios::end);
    const std::streamoff length = in.tellg();
    if (length < 0) {
        return {};
    }
    in.seekg(0, std::ios::beg);
    std::string bytes(static_cast<std::size_t>(length), '\0');
    if (length > 0) {
        in.read(bytes.data(), length);
        if (!in) {
            return {};
        }
    }
    return sha256Hex(bytes);
}

bool writeText(const fs::path& path, const std::string& text, std::string& error) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) {
        error = "could not create " + path.parent_path().string() + ": " + ec.message();
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        error = "could not write " + path.string();
        return false;
    }
    out << text;
    if (!out) {
        error = "could not write " + path.string();
        return false;
    }
    return true;
}

bool installRetrievalConfig(const fs::path& seed, const fs::path& destination, std::string& error) {
    std::error_code ec;
    if (fs::exists(destination, ec)) {
        fs::permissions(destination, fs::perms::owner_write, fs::perm_options::add, ec);
        fs::remove(destination, ec);
    }
    fs::copy_file(seed, destination, fs::copy_options::overwrite_existing, ec);
    if (ec) {
        error = "could not install retrieval config: " + ec.message();
        return false;
    }
    fs::permissions(
        destination,
        fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read,
        fs::perm_options::replace,
        ec);
    if (ec) {
        error = "could not protect retrieval config: " + ec.message();
        return false;
    }
    if (hashFile(seed) != hashFile(destination)) {
        error = "installed retrieval config does not match the seed";
        return false;
    }
    return true;
}

bool validToken(const std::string& value) {
    if (value.empty() || value.size() > 64) {
        return false;
    }
    for (unsigned char c : value) {
        if (!std::isalnum(c) && c != '-' && c != '_' && c != '.') {
            return false;
        }
    }
    return value != "." && value != "..";
}

bool validCase(const SyntheticCase& item, std::string& error) {
    if (!validToken(item.name) || !validToken(item.evidence_name)) {
        error = "synthetic case name is invalid: " + item.name;
        return false;
    }
    if (item.goal_text.empty() || item.evidence_text.empty()) {
        error = "synthetic case is empty: " + item.name;
        return false;
    }
    return true;
}

bool casesAreSeparated(const std::vector<SyntheticCase>& acquisition,
                       const std::vector<SyntheticCase>& evaluation,
                       std::string& error) {
    if (acquisition.empty() || evaluation.empty()) {
        error = "synthetic pilot requires acquisition and evaluation cases";
        return false;
    }
    std::vector<std::string> acquisition_goals;
    std::vector<std::string> acquisition_evidence;
    for (const SyntheticCase& item : acquisition) {
        if (!validCase(item, error)) {
            return false;
        }
        acquisition_goals.push_back(sha256Hex(item.goal_text));
        acquisition_evidence.push_back(sha256Hex(item.evidence_text));
    }
    for (const SyntheticCase& item : evaluation) {
        if (!validCase(item, error)) {
            return false;
        }
        const std::string goal_hash = sha256Hex(item.goal_text);
        const std::string evidence_hash = sha256Hex(item.evidence_text);
        if (std::find(acquisition_goals.begin(), acquisition_goals.end(), goal_hash)
                != acquisition_goals.end()
            || std::find(acquisition_evidence.begin(), acquisition_evidence.end(), evidence_hash)
                != acquisition_evidence.end()) {
            error = "evaluation repeats acquisition material: " + item.name;
            return false;
        }
    }
    return true;
}

struct EngineRun {
    EngineProcessStatus status;
    std::string output;
};

struct EngineEnvironment {
    SyntheticExecution execution = SyntheticExecution::Mock;
    std::string inference_backend = "llama_cpp";
    std::string inference_base_url = "http://127.0.0.1:8080";
    std::string embed_base_url = "http://127.0.0.1:8081";
};

EngineEnvironment environmentFor(const SyntheticAbRequest& request) {
    EngineEnvironment environment;
    environment.execution = request.execution;
    environment.inference_backend = request.inference_backend;
    environment.inference_base_url = request.inference_base_url;
    environment.embed_base_url = request.embed_base_url;
    return environment;
}

void applyEngineEnvironment(const EngineEnvironment& environment) {
    if (environment.execution == SyntheticExecution::Live) {
        unsetenv("THOTH_TEST_SUITE_DEV");
        setenv("THOTH_INFERENCE_BACKEND", environment.inference_backend.c_str(), 1);
        setenv("THOTH_INFERENCE_BASE_URL", environment.inference_base_url.c_str(), 1);
        setenv("THOTH_EMBED_BASE_URL", environment.embed_base_url.c_str(), 1);
        setenv("THOTH_LLM_TIMEOUT_SECONDS", "900", 1);
        return;
    }
    setenv("THOTH_TEST_SUITE_DEV", "1", 1);
}

EngineRun runEngine(const std::string& binary,
                    const fs::path& workspace,
                    const std::vector<std::string>& args,
                    int timeout_seconds,
                    const EngineEnvironment& environment) {
    EngineRun result;
    int pipes[2];
    if (pipe(pipes) != 0) {
        result.output = "pipe failed";
        return result;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipes[0]);
        close(pipes[1]);
        result.output = "fork failed";
        return result;
    }
    if (pid == 0) {
        setenv("THOTH_WORKSPACE_PATH", workspace.c_str(), 1);
        setenv("THOTH_LOGS_PATH", (workspace / "logs").c_str(), 1);
        applyEngineEnvironment(environment);
        unsetenv("THOTH_LOG_CONFIG");
        unsetenv("THOTH_ALP_ENABLED");
        unsetenv("THOTH_ALP_TX_INDEX");
        dup2(pipes[1], STDOUT_FILENO);
        dup2(pipes[1], STDERR_FILENO);
        close(pipes[0]);
        close(pipes[1]);
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(binary.c_str()));
        for (const std::string& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);
        execv(binary.c_str(), argv.data());
        _exit(127);
    }

    close(pipes[1]);
    const int flags = fcntl(pipes[0], F_GETFL, 0);
    fcntl(pipes[0], F_SETFL, flags | O_NONBLOCK);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
    bool exited = false;
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        pollfd fd{};
        fd.fd = pipes[0];
        fd.events = POLLIN;
        poll(&fd, 1, 100);
        char buf[1024];
        while (true) {
            const ssize_t n = read(pipes[0], buf, sizeof(buf));
            if (n > 0) {
                result.output.append(buf, static_cast<std::size_t>(n));
                if (result.output.size() > 65536) {
                    result.output.erase(0, result.output.size() - 65536);
                }
            } else {
                break;
            }
        }
        if (waitpid(pid, &status, WNOHANG) == pid) {
            exited = true;
            break;
        }
    }
    if (!exited) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        result.status.timed_out = true;
        result.status.exited = true;
        result.status.exit_code = -1;
    } else if (WIFEXITED(status)) {
        result.status.exited = true;
        result.status.exit_code = WEXITSTATUS(status);
    } else {
        result.status.exited = true;
        result.status.exit_code = -1;
    }
    close(pipes[0]);
    return result;
}

bool engineSucceeded(const EngineRun& run, const std::string& marker, std::string& error) {
    if (!run.status.exited || run.status.timed_out || run.status.exit_code != 0
        || run.output.find(marker) == std::string::npos) {
        error = "engine did not exit cleanly: " + run.output;
        return false;
    }
    return true;
}

std::string captureEngineImageDigest(const std::string& image_ref) {
    const std::string allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.:@/-";
    if (image_ref.empty() || image_ref.find_first_not_of(allowed) != std::string::npos) {
        return "unavailable";
    }
    int pipes[2];
    if (pipe(pipes) != 0) {
        return "unavailable";
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipes[0]);
        close(pipes[1]);
        return "unavailable";
    }
    if (pid == 0) {
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        dup2(pipes[1], STDOUT_FILENO);
        close(pipes[0]);
        close(pipes[1]);
        execlp("docker", "docker", "image", "inspect", image_ref.c_str(), "--format", "{{.Id}}",
               static_cast<char*>(nullptr));
        _exit(127);
    }
    close(pipes[1]);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    std::string output;
    bool exited = false;
    int status = 0;
    const int flags = fcntl(pipes[0], F_GETFL, 0);
    fcntl(pipes[0], F_SETFL, flags | O_NONBLOCK);
    while (std::chrono::steady_clock::now() < deadline) {
        pollfd fd{};
        fd.fd = pipes[0];
        fd.events = POLLIN;
        poll(&fd, 1, 100);
        char buf[512];
        while (true) {
            const ssize_t n = read(pipes[0], buf, sizeof(buf));
            if (n > 0) {
                output.append(buf, static_cast<std::size_t>(n));
            } else {
                break;
            }
        }
        if (waitpid(pid, &status, WNOHANG) == pid) {
            exited = true;
            break;
        }
    }
    if (!exited) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        close(pipes[0]);
        return "unavailable";
    }
    close(pipes[0]);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return "unavailable";
    }
    const auto end = output.find_first_of("\r\n");
    if (end != std::string::npos) {
        output.resize(end);
    }
    if (output.empty()) {
        return "unavailable";
    }
    return output;
}

bool sameHashedFiles(std::vector<HashedFile> left, std::vector<HashedFile> right) {
    const auto by_label = [](const HashedFile& a, const HashedFile& b) {
        return a.label < b.label;
    };
    std::sort(left.begin(), left.end(), by_label);
    std::sort(right.begin(), right.end(), by_label);
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (left[i].label != right[i].label || left[i].sha256 != right[i].sha256
            || left[i].sha256.empty()) {
            return false;
        }
    }
    return true;
}

nlohmann::json hashedFilesJson(const std::vector<HashedFile>& files) {
    nlohmann::json array = nlohmann::json::array();
    for (const HashedFile& file : files) {
        array.push_back({{"label", file.label}, {"sha256", file.sha256}});
    }
    return array;
}

nlohmann::json serviceModelsJson(const std::vector<ServiceModelIdentity>& models) {
    nlohmann::json array = nlohmann::json::array();
    for (const ServiceModelIdentity& model : models) {
        array.push_back({
            {"role", model.role},
            {"id", model.id},
            {"n_embd", model.n_embd},
            {"size", model.size},
            {"ftype", model.ftype},
        });
    }
    return array;
}

nlohmann::json identityJson(const ArmIdentity& identity) {
    return {
        {"engine_binary_sha256", identity.engine_binary_sha256},
        {"engine_library_sha256", identity.engine_library_sha256},
        {"engine_image_digest", identity.engine_image_digest},
        {"retrieval_config_sha256", identity.retrieval_config_sha256},
        {"model_files", hashedFilesJson(identity.model_files)},
        {"service_models", serviceModelsJson(identity.service_models)},
        {"evidence_files", hashedFilesJson(identity.evidence_files)},
        {"goal_files", hashedFilesJson(identity.goal_files)},
    };
}

bool prepareWorkspace(const fs::path& workspace,
                      const fs::path& retrieval_seed,
                      std::string& error) {
    std::error_code ec;
    fs::create_directories(workspace / "logs", ec);
    if (ec) {
        error = "could not create workspace: " + ec.message();
        return false;
    }
    return installRetrievalConfig(retrieval_seed, workspace / "retrieval_config.json", error);
}

bool writeCaseFiles(const fs::path& workspace, const SyntheticCase& item, std::string& error) {
    const fs::path evidence = workspace / "evidence" / item.name / item.evidence_name;
    const fs::path goal = workspace / "goals" / (item.name + ".txt");
    return writeText(evidence, item.evidence_text, error)
        && writeText(goal, item.goal_text, error);
}

bool runIndex(const std::string& binary,
              const fs::path& workspace,
              const fs::path& directory,
              int timeout_seconds,
              const EngineEnvironment& environment,
              EngineProcessStatus& status,
              std::string& error) {
    const EngineRun run = runEngine(
        binary, workspace, {"--index-directory", directory.string()}, timeout_seconds, environment);
    status = run.status;
    return engineSucceeded(run, "WORKSPACE_INDEX ", error);
}

bool runGoal(const std::string& binary,
             const fs::path& workspace,
             const SyntheticCase& item,
             int timeout_seconds,
             const EngineEnvironment& environment,
             EngineProcessStatus& status,
             std::string& error) {
    const fs::path evidence = workspace / "evidence" / item.name;
    const fs::path goal = workspace / "goals" / (item.name + ".txt");
    const EngineRun run = runEngine(
        binary,
        workspace,
        {"--fixed-evidence-goal", goal.string(), "--evidence-directory", evidence.string()},
        timeout_seconds,
        environment);
    status = run.status;
    return engineSucceeded(run, "FIXED_EVIDENCE_GOAL COMPLETED", error);
}

bool hashModels(const std::vector<std::string>& paths, std::vector<HashedFile>& out, std::string& error) {
    out.clear();
    for (const std::string& path : paths) {
        const fs::path file(path);
        std::error_code ec;
        if (!fs::is_regular_file(file, ec) || ec) {
            error = "model file is missing: " + path;
            return false;
        }
        HashedFile hashed;
        hashed.label = file.filename().string();
        hashed.sha256 = hashFile(file);
        if (hashed.sha256.empty()) {
            error = "model file could not be hashed: " + path;
            return false;
        }
        out.push_back(hashed);
    }
    return true;
}

void addCaseHashes(const fs::path& workspace,
                   const std::vector<SyntheticCase>& cases,
                   ArmIdentity& identity) {
    for (const SyntheticCase& item : cases) {
        HashedFile goal;
        goal.label = item.name;
        goal.sha256 = hashFile(workspace / "goals" / (item.name + ".txt"));
        identity.goal_files.push_back(goal);
        HashedFile evidence;
        evidence.label = item.name + "/" + item.evidence_name;
        evidence.sha256 = hashFile(workspace / "evidence" / item.name / item.evidence_name);
        identity.evidence_files.push_back(evidence);
    }
}

std::string countsText(const ExperienceTableCounts& counts) {
    std::ostringstream out;
    out << "trajectories=" << counts.trajectories
        << " strategies=" << counts.strategies
        << " past_plans=" << counts.past_plans
        << " graph_edges=" << counts.graph_edges
        << " facts=" << counts.facts;
    return out.str();
}

struct HttpEndpoint {
    std::string host;
    std::string port;
};

bool parseHttpEndpoint(const std::string& url, HttpEndpoint& endpoint, std::string& error) {
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) {
        error = "inference URL must use http: " + url;
        return false;
    }
    std::string rest = url.substr(prefix.size());
    while (!rest.empty() && rest.back() == '/') {
        rest.pop_back();
    }
    if (rest.empty() || rest.find('/') != std::string::npos) {
        error = "inference URL must be a host and port: " + url;
        return false;
    }
    const auto colon = rest.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= rest.size()) {
        error = "inference URL is missing a port: " + url;
        return false;
    }
    endpoint.host = rest.substr(0, colon);
    endpoint.port = rest.substr(colon + 1);
    if (endpoint.host.size() >= 2 && endpoint.host.front() == '[' && endpoint.host.back() == ']') {
        endpoint.host = endpoint.host.substr(1, endpoint.host.size() - 2);
    }
    if (endpoint.host.empty() || endpoint.port.empty()) {
        error = "inference URL is missing a port: " + url;
        return false;
    }
    return true;
}

bool httpGet(const std::string& base_url, const std::string& path, std::string& body, std::string& error) {
    body.clear();
    HttpEndpoint endpoint;
    if (!parseHttpEndpoint(base_url, endpoint, error)) {
        return false;
    }
    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    addrinfo* addresses = nullptr;
    if (getaddrinfo(endpoint.host.c_str(), endpoint.port.c_str(), &hints, &addresses) != 0
        || addresses == nullptr) {
        error = "inference service is unreachable: " + base_url;
        return false;
    }
    int fd = -1;
    for (addrinfo* address = addresses; address != nullptr; address = address->ai_next) {
        fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) {
            continue;
        }
        const int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        const int connected = connect(fd, address->ai_addr, address->ai_addrlen);
        if (connected < 0 && errno != EINPROGRESS) {
            close(fd);
            fd = -1;
            continue;
        }
        pollfd ready{};
        ready.fd = fd;
        ready.events = POLLOUT;
        if (poll(&ready, 1, 3000) <= 0) {
            close(fd);
            fd = -1;
            continue;
        }
        int so_error = 0;
        socklen_t length = sizeof(so_error);
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &length) != 0 || so_error != 0) {
            close(fd);
            fd = -1;
            continue;
        }
        fcntl(fd, F_SETFL, flags);
        break;
    }
    freeaddrinfo(addresses);
    if (fd < 0) {
        error = "inference service is unreachable: " + base_url;
        return false;
    }
    const std::string request = "GET " + path + " HTTP/1.1\r\nHost: " + endpoint.host + ":"
        + endpoint.port + "\r\nConnection: close\r\n\r\n";
    std::size_t sent = 0;
    while (sent < request.size()) {
        const ssize_t n = send(fd, request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            close(fd);
            error = "inference service is unreachable: " + base_url;
            return false;
        }
        sent += static_cast<std::size_t>(n);
    }
    std::string raw;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline && raw.size() < 1024 * 1024) {
        pollfd ready{};
        ready.fd = fd;
        ready.events = POLLIN;
        if (poll(&ready, 1, 200) <= 0) {
            continue;
        }
        char buffer[4096];
        const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n == 0) {
            break;
        }
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            break;
        }
        raw.append(buffer, static_cast<std::size_t>(n));
        const auto header_end = raw.find("\r\n\r\n");
        const auto length_at = raw.find("Content-Length:");
        if (header_end != std::string::npos && length_at != std::string::npos && length_at < header_end) {
            const auto value_at = length_at + std::string("Content-Length:").size();
            const auto line_end = raw.find("\r\n", value_at);
            if (line_end != std::string::npos) {
                try {
                    const std::size_t length = static_cast<std::size_t>(
                        std::stoul(raw.substr(value_at, line_end - value_at)));
                    if (raw.size() >= header_end + 4 + length) {
                        break;
                    }
                } catch (const std::exception&) {
                    break;
                }
            }
        }
    }
    close(fd);
    const auto header_end = raw.find("\r\n\r\n");
    const auto status_end = raw.find("\r\n");
    const std::string status = status_end == std::string::npos ? std::string() : raw.substr(0, status_end);
    if (header_end == std::string::npos || status.rfind("HTTP/", 0) != 0 || status.find(" 200") == std::string::npos) {
        error = "inference service is unreachable: " + base_url;
        return false;
    }
    body = raw.substr(header_end + 4);
    const std::string marker = "Content-Length:";
    const auto length_at = raw.find(marker);
    if (length_at != std::string::npos && length_at < header_end) {
        const auto value_at = length_at + marker.size();
        const auto line_end = raw.find("\r\n", value_at);
        try {
            const std::size_t length = static_cast<std::size_t>(
                std::stoul(raw.substr(value_at, line_end - value_at)));
            if (body.size() < length) {
                error = "inference service is unreachable: " + base_url;
                return false;
            }
            body.resize(length);
        } catch (const std::exception&) {
            error = "inference service is unreachable: " + base_url;
            return false;
        }
    }
    if (body.empty()) {
        error = "inference service is unreachable: " + base_url;
        return false;
    }
    return true;
}

bool parseServiceModel(const std::string& role,
                       const std::string& body,
                       ServiceModelIdentity& model,
                       std::string& error) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(body);
    } catch (const std::exception&) {
        error = role + " service model response is not JSON";
        return false;
    }
    if (!json.contains("data") || !json["data"].is_array() || json["data"].size() != 1
        || !json["data"][0].is_object()) {
        error = role + " service did not report exactly one model";
        return false;
    }
    const nlohmann::json& item = json["data"][0];
    if (!item.contains("meta") || !item["meta"].is_object()) {
        error = role + " service model metadata is missing";
        return false;
    }
    try {
        model.role = role;
        model.id = item.at("id").get<std::string>();
        model.n_embd = item["meta"].at("n_embd").get<int>();
        model.size = item["meta"].at("size").get<std::int64_t>();
        model.ftype = item["meta"].at("ftype").get<std::string>();
    } catch (const std::exception&) {
        error = role + " service model metadata is incomplete";
        return false;
    }
    if (model.id.empty() || model.ftype.empty() || model.n_embd <= 0 || model.size <= 0) {
        error = role + " service model metadata is incomplete";
        return false;
    }
    if (role == "embed") {
        const EmbeddingEngine engine(EmbeddingEngine::Method::External);
        if (model.n_embd != engine.getDimension()) {
            error = "embedding service dimension does not match the engine";
            return false;
        }
    }
    error.clear();
    return true;
}

bool readServiceModels(const SyntheticAbRequest& request,
                       std::vector<ServiceModelIdentity>& models,
                       std::string& error) {
    models.clear();
    std::string chat_body;
    std::string embed_body;
    if (!httpGet(request.inference_base_url, "/v1/models", chat_body, error)
        || !httpGet(request.embed_base_url, "/v1/models", embed_body, error)) {
        return false;
    }
    ServiceModelIdentity chat;
    ServiceModelIdentity embed;
    if (!parseServiceModel("chat", chat_body, chat, error)
        || !parseServiceModel("embed", embed_body, embed, error)) {
        return false;
    }
    models.push_back(chat);
    models.push_back(embed);
    return true;
}

std::string traceLoadedObjects(const std::string& engine_binary, bool& exited_cleanly) {
    exited_cleanly = false;
    int pipes[2];
    if (pipe(pipes) != 0) {
        return {};
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(pipes[0]);
        close(pipes[1]);
        return {};
    }
    if (pid == 0) {
        dup2(pipes[1], STDOUT_FILENO);
        dup2(pipes[1], STDERR_FILENO);
        close(pipes[0]);
        close(pipes[1]);
        setenv("LD_TRACE_LOADED_OBJECTS", "1", 1);
        char* argv[] = {const_cast<char*>(engine_binary.c_str()), nullptr};
        execv(engine_binary.c_str(), argv);
        _exit(127);
    }
    close(pipes[1]);
    const int flags = fcntl(pipes[0], F_GETFL, 0);
    fcntl(pipes[0], F_SETFL, flags | O_NONBLOCK);
    std::string output;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool exited = false;
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        pollfd fd{};
        fd.fd = pipes[0];
        fd.events = POLLIN;
        poll(&fd, 1, 100);
        char buf[1024];
        while (true) {
            const ssize_t n = read(pipes[0], buf, sizeof(buf));
            if (n > 0) {
                output.append(buf, static_cast<std::size_t>(n));
            } else {
                break;
            }
        }
        if (waitpid(pid, &status, WNOHANG) == pid) {
            exited = true;
            break;
        }
    }
    if (!exited) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
    close(pipes[0]);
    exited_cleanly = exited && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    return output;
}

} // namespace

bool armIdentitiesMatch(const ArmIdentity& arm_a,
                         const ArmIdentity& arm_b,
                         std::string& error) {
    if (arm_a.engine_image_digest.empty() || arm_a.engine_image_digest != arm_b.engine_image_digest) {
        error = "engine image digest mismatch";
        return false;
    }
    if (arm_a.engine_binary_sha256.empty()
        || arm_a.engine_binary_sha256 != arm_b.engine_binary_sha256) {
        error = "engine binary mismatch";
        return false;
    }
    if (arm_a.retrieval_config_sha256.empty()
        || arm_a.retrieval_config_sha256 != arm_b.retrieval_config_sha256) {
        error = "retrieval configuration mismatch";
        return false;
    }
    if (!sameHashedFiles(arm_a.model_files, arm_b.model_files)) {
        error = "model hash mismatch";
        return false;
    }
    if (!sameHashedFiles(arm_a.evidence_files, arm_b.evidence_files)) {
        error = "evidence hash mismatch";
        return false;
    }
    if (!sameHashedFiles(arm_a.goal_files, arm_b.goal_files)) {
        error = "goal hash mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool liveArmIdentitiesMatch(const ArmIdentity& arm_a,
                             const ArmIdentity& arm_b,
                             bool require_engine_image,
                             std::string& error) {
    if (arm_a.engine_library_sha256.size() != 64
        || arm_a.engine_library_sha256 != arm_b.engine_library_sha256) {
        error = "engine library mismatch";
        return false;
    }
    if (arm_a.service_models.size() != 2 || arm_a.service_models != arm_b.service_models) {
        error = "service model identity mismatch";
        return false;
    }
    bool saw_chat = false;
    bool saw_embed = false;
    for (const ServiceModelIdentity& model : arm_a.service_models) {
        if (model.id.empty() || model.ftype.empty() || model.n_embd <= 0 || model.size <= 0) {
            error = "service model identity mismatch";
            return false;
        }
        if (model.role == "chat") {
            saw_chat = true;
        } else if (model.role == "embed") {
            saw_embed = true;
            const EmbeddingEngine engine(EmbeddingEngine::Method::External);
            if (model.n_embd != engine.getDimension()) {
                error = "embedding service dimension does not match the engine";
                return false;
            }
        } else {
            error = "service model identity mismatch";
            return false;
        }
    }
    if (!saw_chat || !saw_embed) {
        error = "service model identity mismatch";
        return false;
    }
    const bool image_missing = arm_a.engine_image_digest.empty()
        || arm_a.engine_image_digest == "unavailable"
        || arm_b.engine_image_digest.empty()
        || arm_b.engine_image_digest == "unavailable";
    if (require_engine_image && image_missing) {
        error = "engine image digest unavailable";
        return false;
    }
    if (!image_missing && arm_a.engine_image_digest != arm_b.engine_image_digest) {
        error = "engine image digest mismatch";
        return false;
    }
    ArmIdentity compared_a = arm_a;
    ArmIdentity compared_b = arm_b;
    compared_a.model_files.clear();
    compared_b.model_files.clear();
    if (image_missing) {
        compared_a.engine_image_digest = "not-required";
        compared_b.engine_image_digest = "not-required";
    }
    return armIdentitiesMatch(compared_a, compared_b, error);
}

bool resolveLoadedAgentLibrary(const std::string& engine_binary,
                               std::string& library_path,
                               std::string& error) {
    library_path.clear();
    bool exited_cleanly = false;
    const std::string output = traceLoadedObjects(engine_binary, exited_cleanly);
    if (!exited_cleanly) {
        error = "loaded agent library was not found";
        return false;
    }
    int matches = 0;
    std::string found;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        const auto arrow = line.find("=>");
        const auto name = line.find("libbasic_agent.so");
        if (arrow == std::string::npos || name == std::string::npos || name > arrow) {
            continue;
        }
        ++matches;
        std::string rest = line.substr(arrow + 2);
        while (!rest.empty() && std::isspace(static_cast<unsigned char>(rest.front()))) {
            rest.erase(rest.begin());
        }
        const auto address = rest.find(" (");
        if (address != std::string::npos) {
            rest.resize(address);
        }
        while (!rest.empty() && std::isspace(static_cast<unsigned char>(rest.back()))) {
            rest.pop_back();
        }
        found = rest;
    }
    if (matches > 1) {
        error = "loaded agent library is ambiguous";
        return false;
    }
    if (matches != 1 || found.empty() || found == "not found" || found.front() != '/') {
        error = "loaded agent library was not found";
        return false;
    }
    std::error_code ec;
    if (!fs::is_regular_file(found, ec) || ec) {
        error = "loaded agent library is not a file";
        return false;
    }
    library_path = found;
    error.clear();
    return true;
}

bool snapshotAfterEngineExit(const EngineProcessStatus& status,
                             const std::string& source_database,
                             const std::string& snapshot_directory,
                             const ExperienceTableCounts* required_counts,
                             MemorySnapshot& snapshot,
                             std::string& error) {
    snapshot = MemorySnapshot{};
    if (!status.exited || status.timed_out) {
        error = "engine has not exited; refusing to snapshot";
        return false;
    }
    if (status.exit_code != 0) {
        error = "engine exit failed; refusing to snapshot";
        return false;
    }
    return createConsistentMemorySnapshot(
        source_database, snapshot_directory, required_counts, snapshot, error);
}

bool runSyntheticAbPilot(const SyntheticAbRequest& request,
                         SyntheticAbReport& report,
                         std::string& error) {
    report = SyntheticAbReport{};
    if (!casesAreSeparated(request.acquisition, request.evaluation, error)) {
        return false;
    }
    std::error_code ec;
    const fs::path binary(request.engine_binary);
    if (!fs::is_regular_file(binary, ec) || ec) {
        error = "engine binary is missing: " + request.engine_binary;
        return false;
    }
    const fs::path retrieval_seed(request.retrieval_config_path);
    if (!fs::is_regular_file(retrieval_seed, ec) || ec) {
        error = "retrieval config is missing: " + request.retrieval_config_path;
        return false;
    }
    const std::string retrieval_hash = hashFile(retrieval_seed);
    const std::string binary_hash = hashFile(binary);
    if (retrieval_hash.empty() || binary_hash.empty()) {
        error = "pilot identity files could not be hashed";
        return false;
    }
    std::vector<HashedFile> model_hashes;
    if (!hashModels(request.model_files, model_hashes, error)) {
        return false;
    }
    const bool live = request.execution == SyntheticExecution::Live;
    report.execution = live ? "live" : "mock";
    const EngineEnvironment environment = environmentFor(request);
    std::string library_path;
    std::string library_hash;
    std::vector<ServiceModelIdentity> service_models;
    if (live) {
        if (request.inference_backend.empty() || request.inference_base_url.empty()
            || request.embed_base_url.empty()) {
            error = "live inference endpoint is missing";
            return false;
        }
        if (!resolveLoadedAgentLibrary(request.engine_binary, library_path, error)) {
            return false;
        }
        library_hash = hashFile(library_path);
        if (library_hash.empty()) {
            error = "loaded agent library could not be hashed";
            return false;
        }
        if (!readServiceModels(request, service_models, error)) {
            return false;
        }
    }
    const std::string image_digest = captureEngineImageDigest(request.engine_image);

    const fs::path root(request.work_root);
    fs::create_directories(root, ec);
    if (ec) {
        error = "pilot workspace could not be created: " + ec.message();
        return false;
    }
    const fs::path acquisition = root / "acquisition";
    const fs::path frozen_dir = root / "frozen";
    const fs::path init_workspace = root / "arm_a_init";
    if (!prepareWorkspace(acquisition, retrieval_seed, error)
        || !prepareWorkspace(init_workspace, retrieval_seed, error)) {
        return false;
    }
    fs::create_directories(init_workspace / "empty", ec);

    const int timeout = request.engine_timeout_seconds;
    EngineProcessStatus last_acquisition;
    for (const SyntheticCase& item : request.acquisition) {
        if (!writeCaseFiles(acquisition, item, error)) {
            return false;
        }
        EngineProcessStatus index_status;
        if (!runIndex(request.engine_binary, acquisition,
                      acquisition / "evidence" / item.name, timeout, environment, index_status, error)) {
            return false;
        }
        if (!runGoal(request.engine_binary, acquisition, item, timeout, environment, last_acquisition, error)) {
            return false;
        }
    }

    MemorySnapshot acquisition_snapshot;
    if (!snapshotAfterEngineExit(
            last_acquisition,
            (acquisition / "memory.db").string(),
            frozen_dir.string(),
            nullptr,
            acquisition_snapshot,
            error)) {
        return false;
    }
    const fs::path acquisition_verify = root / "verify_acquisition" / "memory.db";
    if (!restoreMemorySnapshot(acquisition_snapshot.database_path, acquisition_verify.string(), error)) {
        return false;
    }
    if (!readExperienceTableCounts(acquisition_verify.string(), report.acquisition_counts, error)) {
        return false;
    }
    const long long expected = static_cast<long long>(request.acquisition.size());
    if (report.acquisition_counts.trajectories != expected
        || report.acquisition_counts.past_plans != expected) {
        error = "acquisition did not preserve one trajectory and past plan per goal: "
            + countsText(report.acquisition_counts);
        std::error_code remove_ec;
        fs::remove(acquisition_snapshot.database_path, remove_ec);
        return false;
    }
    report.acquisition_sha256 = acquisition_snapshot.sha256;

    EngineProcessStatus init_status;
    if (!runIndex(request.engine_binary, init_workspace, (init_workspace / "empty").string(),
                  timeout, environment, init_status, error)) {
        return false;
    }
    const ExperienceTableCounts empty_counts;
    MemorySnapshot empty_snapshot;
    if (!snapshotAfterEngineExit(
            init_status,
            (init_workspace / "memory.db").string(),
            (root / "empty_template").string(),
            &empty_counts,
            empty_snapshot,
            error)) {
        return false;
    }
    report.empty_experience_sha256 = empty_snapshot.sha256;

    const fs::path frozen_file = frozen_dir / "memory.db";
    const std::string frozen_before = hashFile(frozen_file);
    if (frozen_before != report.acquisition_sha256) {
        error = "frozen snapshot hash changed before evaluation";
        return false;
    }

    for (const SyntheticCase& item : request.evaluation) {
        const fs::path arm_a = root / "arm_a" / item.name;
        const fs::path arm_b = root / "arm_b" / item.name;
        if (arm_a == arm_b || arm_a == acquisition || arm_b == acquisition
            || arm_a == frozen_dir || arm_b == frozen_dir) {
            error = "evaluation workspace is not isolated: " + item.name;
            return false;
        }
        if (!prepareWorkspace(arm_a, retrieval_seed, error)
            || !prepareWorkspace(arm_b, retrieval_seed, error)) {
            return false;
        }
        if (!restoreMemorySnapshot(empty_snapshot.database_path, (arm_a / "memory.db").string(), error)
            || !restoreMemorySnapshot(frozen_file.string(), (arm_b / "memory.db").string(), error)) {
            return false;
        }
        if (!writeCaseFiles(arm_a, item, error) || !writeCaseFiles(arm_b, item, error)) {
            return false;
        }

        EngineProcessStatus arm_a_index;
        EngineProcessStatus arm_b_index;
        if (!runIndex(request.engine_binary, arm_a, arm_a / "evidence" / item.name,
                      timeout, environment, arm_a_index, error)
            || !runIndex(request.engine_binary, arm_b, arm_b / "evidence" / item.name,
                         timeout, environment, arm_b_index, error)) {
            return false;
        }

        MemorySnapshot arm_a_check;
        MemorySnapshot arm_b_check;
        if (!snapshotAfterEngineExit(
                arm_a_index, (arm_a / "memory.db").string(),
                (root / "pre_goal" / "arm_a" / item.name).string(),
                &empty_counts, arm_a_check, error)) {
            error = "Arm A experience was not empty before the goal: " + error;
            return false;
        }
        if (!snapshotAfterEngineExit(
                arm_b_index, (arm_b / "memory.db").string(),
                (root / "pre_goal" / "arm_b" / item.name).string(),
                &report.acquisition_counts, arm_b_check, error)) {
            error = "Arm B copy did not preserve the acquisition snapshot: " + error;
            return false;
        }
        if (hashFile(frozen_file) != frozen_before) {
            error = "frozen snapshot changed during evaluation";
            return false;
        }

        EngineProcessStatus arm_a_goal;
        EngineProcessStatus arm_b_goal;
        if (!runGoal(request.engine_binary, arm_a, item, timeout, environment, arm_a_goal, error)
            || !runGoal(request.engine_binary, arm_b, item, timeout, environment, arm_b_goal, error)) {
            return false;
        }
        if (hashFile(frozen_file) != frozen_before) {
            error = "frozen snapshot changed during evaluation";
            return false;
        }
    }

    const fs::path untouched = root / "untouched_restore" / "memory.db";
    if (!restoreMemorySnapshot(frozen_file.string(), untouched.string(), error)) {
        return false;
    }
    report.frozen_sha256_after_evaluation = hashFile(frozen_file);
    report.frozen_unchanged = report.frozen_sha256_after_evaluation == report.acquisition_sha256
        && hashFile(untouched) == report.acquisition_sha256
        && !fs::exists(fs::path(frozen_file.string() + "-wal"));
    if (!report.frozen_unchanged) {
        error = "frozen snapshot changed during evaluation";
        return false;
    }
    if (hashFile(binary) != binary_hash || hashFile(retrieval_seed) != retrieval_hash) {
        error = "pilot identity files changed during evaluation";
        return false;
    }
    std::vector<HashedFile> models_after;
    if (!hashModels(request.model_files, models_after, error)
        || !sameHashedFiles(model_hashes, models_after)) {
        error = "model files changed during evaluation";
        return false;
    }
    if (live) {
        std::string library_after;
        if (!resolveLoadedAgentLibrary(request.engine_binary, library_after, error)
            || hashFile(library_after) != library_hash) {
            error = "loaded agent library changed during evaluation";
            return false;
        }
        std::vector<ServiceModelIdentity> models_after_service;
        if (!readServiceModels(request, models_after_service, error)
            || models_after_service != service_models) {
            error = "service model identity changed during evaluation";
            return false;
        }
    }

    const fs::path first_a = root / "arm_a" / request.evaluation.front().name;
    const fs::path first_b = root / "arm_b" / request.evaluation.front().name;
    report.arm_a.engine_binary_sha256 = binary_hash;
    report.arm_b.engine_binary_sha256 = binary_hash;
    report.arm_a.engine_library_sha256 = library_hash;
    report.arm_b.engine_library_sha256 = library_hash;
    report.engine_library_path = library_path;
    report.engine_library_sha256 = library_hash;
    report.arm_a.service_models = service_models;
    report.arm_b.service_models = service_models;
    report.service_models = service_models;
    report.arm_a.engine_image_digest = image_digest;
    report.arm_b.engine_image_digest = image_digest;
    report.engine_image_digest = image_digest;
    report.engine_image_verified = !image_digest.empty() && image_digest != "unavailable";
    report.arm_a.model_files = model_hashes;
    report.arm_b.model_files = model_hashes;
    report.arm_a.retrieval_config_sha256 = hashFile(first_a / "retrieval_config.json");
    report.arm_b.retrieval_config_sha256 = hashFile(first_b / "retrieval_config.json");
    for (const SyntheticCase& item : request.evaluation) {
        addCaseHashes(root / "arm_a" / item.name, {item}, report.arm_a);
        addCaseHashes(root / "arm_b" / item.name, {item}, report.arm_b);
    }
    if (report.arm_a.retrieval_config_sha256 != retrieval_hash
        || report.arm_b.retrieval_config_sha256 != retrieval_hash) {
        error = "retrieval configuration mismatch";
        return false;
    }
    report.identities_match = live
        ? liveArmIdentitiesMatch(report.arm_a, report.arm_b, request.require_engine_image, error)
        : armIdentitiesMatch(report.arm_a, report.arm_b, error);
    if (!report.identities_match) {
        return false;
    }

    nlohmann::json body = {
        {"execution", report.execution},
        {"acquisition_sha256", report.acquisition_sha256},
        {"acquisition_counts",
         {{"trajectories", report.acquisition_counts.trajectories},
          {"strategies", report.acquisition_counts.strategies},
          {"past_plans", report.acquisition_counts.past_plans},
          {"graph_edges", report.acquisition_counts.graph_edges},
          {"facts", report.acquisition_counts.facts}}},
        {"empty_experience_sha256", report.empty_experience_sha256},
        {"frozen_sha256_after_evaluation", report.frozen_sha256_after_evaluation},
        {"frozen_unchanged", report.frozen_unchanged},
        {"identities_match", report.identities_match},
        {"engine_image_digest", report.engine_image_digest},
        {"engine_image_verified", report.engine_image_verified},
        {"engine_library_path", report.engine_library_path},
        {"engine_library_sha256", report.engine_library_sha256},
        {"service_models", serviceModelsJson(report.service_models)},
        {"arm_a", identityJson(report.arm_a)},
        {"arm_b", identityJson(report.arm_b)},
    };
    report.report_path = (root / "ab_report.json").string();
    if (!writeText(report.report_path, body.dump(2) + "\n", error)) {
        return false;
    }
    error.clear();
    return true;
}

} // namespace Thoth
