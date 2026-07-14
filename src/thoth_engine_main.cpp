/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — headless engine entry point (Plan C / Plan F HTTP)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/engine_http_transport.h"
#include "../include/engine_runtime.h"
#include "../include/runtime_bootstrap.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <optional>
#include <pthread.h>
#include <string>
#include <thread>

#ifndef THOTH_ENGINE_VERSION
#define THOTH_ENGINE_VERSION "0.2"
#endif

namespace {

volatile sig_atomic_t g_stop_requested = 0;

void onSignal(int) {
    g_stop_requested = 1;
}

void installShutdownSignals() {
    struct sigaction action {};
    action.sa_handler = onSignal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
}

void blockShutdownSignals(sigset_t& out_mask) {
    sigemptyset(&out_mask);
    sigaddset(&out_mask, SIGINT);
    sigaddset(&out_mask, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &out_mask, nullptr);
}

bool waitForShutdownSignal(const sigset_t& mask,
                           std::chrono::milliseconds timeout) {
    timespec ts {};
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) {
        return false;
    }

    const auto timeout_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout).count();
    ts.tv_sec += static_cast<time_t>(timeout_ns / 1'000'000'000LL);
    ts.tv_nsec += static_cast<long>(timeout_ns % 1'000'000'000LL);
    if (ts.tv_nsec >= 1'000'000'000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1'000'000'000L;
    }

    siginfo_t info {};
    const int sig = sigtimedwait(&mask, &info, &ts);
    return sig == SIGINT || sig == SIGTERM;
}

bool stopRequested() {
    return g_stop_requested != 0;
}

void printUsage(std::ostream& out) {
    out << "Usage: thoth-engine [--help] [--version]\n"
        << "       thoth-engine --execute \"<prompt>\"\n"
        << "       thoth-engine --serve [--bind HOST] [--port PORT]\n"
        << "       thoth-engine\n"
        << "\n"
        << "Modes:\n"
        << "  (default)   Read prompts from stdin until EOF or SIGINT\n"
        << "  --execute   Process one prompt and exit\n"
        << "  --serve     Run HTTP API server (Plan F)\n"
        << "\n"
        << "HTTP server options (--serve):\n"
        << "  --bind HOST   Bind address (default: 127.0.0.1)\n"
        << "  --port PORT   Listen port (default: 8090)\n"
        << "\n"
        << "Environment (see docs/GETTING_STARTED.md):\n"
        << "  THOTH_WORKSPACE_PATH, THOTH_LOGS_PATH, THOTH_INFERENCE_BASE_URL\n"
        << "  THOTH_INFERENCE_BACKEND      ollama | llama_cpp (default: ollama)\n"
        << "  THOTH_ENGINE_BIND, THOTH_ENGINE_PORT\n"
        << "  THOTH_LOG_CONFIG=1          Print resolved paths at startup\n";
}

struct CliOptions {
    bool showHelp = false;
    bool showVersion = false;
    bool serve = false;
    std::optional<std::string> executePrompt;
    Thoth::EngineHttpConfig http_config = Thoth::engineHttpConfigFromEnvironment();
    bool error = false;
};

bool parsePort(const std::string& value, int& port) {
    try {
        port = std::stoi(value);
        return port > 0 && port <= 65535;
    } catch (...) {
        return false;
    }
}

CliOptions parseArgs(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.showHelp = true;
            return options;
        }
        if (arg == "--version") {
            options.showVersion = true;
            return options;
        }
        if (arg == "--serve") {
            options.serve = true;
            continue;
        }
        if (arg == "--bind") {
            if (i + 1 >= argc) {
                std::cerr << "thoth-engine: --bind requires a host argument\n";
                options.error = true;
                return options;
            }
            options.http_config.bind_host = argv[++i];
            continue;
        }
        if (arg == "--port") {
            if (i + 1 >= argc) {
                std::cerr << "thoth-engine: --port requires a port argument\n";
                options.error = true;
                return options;
            }
            if (!parsePort(argv[++i], options.http_config.port)) {
                std::cerr << "thoth-engine: invalid --port value\n";
                options.error = true;
                return options;
            }
            continue;
        }
        if (arg == "--execute") {
            if (i + 1 >= argc) {
                std::cerr << "thoth-engine: --execute requires a prompt argument\n";
                options.error = true;
                return options;
            }
            options.executePrompt = argv[++i];
            continue;
        }

        std::cerr << "thoth-engine: unknown argument: " << arg << '\n';
        options.error = true;
        return options;
    }

    if (options.serve && options.executePrompt) {
        std::cerr << "thoth-engine: --serve cannot be combined with --execute\n";
        options.error = true;
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    Thoth::bootstrapRuntimeEnvironment();

    const CliOptions options = parseArgs(argc, argv);
    if (options.error) {
        printUsage(std::cerr);
        return 1;
    }
    if (options.showHelp) {
        Thoth::logResolvedRuntimeConfig(nullptr);
        printUsage(std::cout);
        return 0;
    }
    if (options.showVersion) {
        Thoth::logResolvedRuntimeConfig(nullptr);
        std::cout << "thoth-engine " << THOTH_ENGINE_VERSION << '\n';
        return 0;
    }

    if (!options.executePrompt && !options.serve) {
        installShutdownSignals();
    }

    sigset_t shutdown_signals {};
    if (options.serve) {
        blockShutdownSignals(shutdown_signals);
    }

    try {
        auto runtime = Thoth::EngineRuntime::create();

        if (options.serve) {
            Thoth::EngineHttpTransport transport(*runtime, options.http_config);

            std::cerr << "[thoth-engine] serving http://" << options.http_config.bind_host << ':'
                      << options.http_config.port << std::endl;

            std::atomic<bool> bind_failed{false};
            std::thread server_thread([&]() {
                if (!transport.run()) {
                    bind_failed = true;
                }
            });

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            if (bind_failed.load()) {
                server_thread.join();
                std::cerr << "[thoth-engine] failed to bind http://"
                          << options.http_config.bind_host << ':'
                          << options.http_config.port << std::endl;
                runtime->shutdown();
                return 1;
            }

            while (!bind_failed.load()) {
                if (waitForShutdownSignal(shutdown_signals,
                                          std::chrono::milliseconds(50))) {
                    break;
                }
            }

            transport.requestStop();
            server_thread.join();

            std::cerr << "[thoth-engine] shutting down" << std::endl;
            runtime->shutdown();
            std::cerr << "[thoth-engine] shutdown complete" << std::endl;
            return 0;
        }

        if (options.executePrompt) {
            std::cout << runtime->submitChat("default", *options.executePrompt).get() << '\n';
            runtime->shutdown();
            return 0;
        }

        std::string line;
        while (!stopRequested() && std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            std::cout << runtime->submitChat("default", line).get() << '\n';
        }

        runtime->shutdown();
    } catch (const std::exception& e) {
        std::cerr << "[thoth-engine] fatal: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "[thoth-engine] fatal: unknown error\n";
        return 1;
    }

    return 0;
}
