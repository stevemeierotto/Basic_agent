/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — headless engine entry point (Plan C / Plan F HTTP)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/agent_context_retrieval.h"
#include "../include/engine_http_transport.h"
#include "../include/engine_runtime.h"
#include "../include/fixed_evidence_goal_run.h"
#include "../include/runtime_bootstrap.h"
#include "../include/workspace_index_run.h"

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
        << "       thoth-engine --fixed-evidence-goal PATH --evidence-directory DIR\n"
        << "       thoth-engine --index-directory PATH\n"
        << "       thoth-engine --serve [--bind HOST] [--port PORT]\n"
        << "       thoth-engine\n"
        << "\n"
        << "Modes:\n"
        << "  (default)                 Read prompts from stdin until EOF or SIGINT\n"
        << "  --execute                 Process one chat prompt and exit\n"
        << "  --fixed-evidence-goal     Read one local goal file, run\n"
        << "                            execute_fixed_evidence_goal, and exit\n"
        << "  --evidence-directory      Authorized evidence directory for that goal\n"
        << "  --index-directory         Index one workspace directory, save\n"
        << "                            rag_index.bin, and exit\n"
        << "  --serve                   Run HTTP API server (Plan F)\n"
        << "\n"
        << "Fixed-evidence goal:\n"
        << "  PATH is one local file. DIR is an existing directory; a symlink to a\n"
        << "  directory is resolved. legacy_orphan chunks are admitted only under\n"
        << "  that directory. An absent or invalid DIR exits before the engine\n"
        << "  starts. The process uses THOTH_WORKSPACE_PATH and the existing\n"
        << "  inference environment, writes the usual workspace logs, and exits.\n"
        << "  One goal per process. Do not combine with --serve or --execute.\n"
        << "  A one-shot container attaches to an existing llama network. Example:\n"
        << "    docker run --rm --network <llama-network> \\\n"
        << "      -e THOTH_WORKSPACE_PATH=/workspace -e THOTH_LOGS_PATH=/logs \\\n"
        << "      -e THOTH_INFERENCE_BACKEND=llama_cpp \\\n"
        << "      -e THOTH_INFERENCE_BASE_URL=http://llama-server:8080 \\\n"
        << "      -e THOTH_EMBED_BASE_URL=http://llama-embed-server:8081 \\\n"
        << "      -e THOTH_LLM_TIMEOUT_SECONDS=900 \\\n"
        << "      -v <workspace>:/workspace -v <goal-file>:/goal.txt:ro \\\n"
        << "      -v <evidence>:/evidence:ro \\\n"
        << "      thoth-engine:local --fixed-evidence-goal /goal.txt --evidence-directory /evidence\n"
        << "\n"
        << "Workspace index:\n"
        << "  PATH must be a directory inside THOTH_WORKSPACE_PATH. The resolved\n"
        << "  directory and every symlink target under it must stay inside that\n"
        << "  workspace. The process loads the existing GRAG index, indexes PATH,\n"
        << "  saves rag_index.bin, and exits. It does not run a goal.\n"
        << "\n"
        << "HTTP server options (--serve):\n"
        << "  --bind HOST   Bind address (default: 127.0.0.1)\n"
        << "  --port PORT   Listen port (default: 8090)\n"
        << "\n"
        << "Environment (see docs/GETTING_STARTED.md):\n"
        << "  THOTH_WORKSPACE_PATH, THOTH_LOGS_PATH\n"
        << "  THOTH_INFERENCE_BASE_URL     llama.cpp chat (default: http://127.0.0.1:8080)\n"
        << "  THOTH_EMBED_BASE_URL         llama.cpp embeddings (default: http://127.0.0.1:8081)\n"
        << "  THOTH_INFERENCE_BACKEND      llama_cpp | ollama (default: llama_cpp)\n"
        << "  THOTH_ENGINE_BIND, THOTH_ENGINE_PORT\n"
        << "  THOTH_LOG_CONFIG=1          Print resolved paths at startup\n";
}

struct CliOptions {
    bool showHelp = false;
    bool showVersion = false;
    bool serve = false;
    std::optional<std::string> executePrompt;
    std::optional<std::string> fixedEvidenceGoal;
    std::optional<std::string> evidenceDirectory;
    std::optional<std::string> indexDirectory;
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
        if (arg == "--fixed-evidence-goal") {
            if (options.fixedEvidenceGoal) {
                std::cerr << "thoth-engine: one goal file per process\n";
                options.error = true;
                return options;
            }
            if (i + 1 >= argc) {
                std::cerr << "thoth-engine: --fixed-evidence-goal requires a file path\n";
                options.error = true;
                return options;
            }
            options.fixedEvidenceGoal = argv[++i];
            continue;
        }
        if (arg == "--evidence-directory") {
            if (options.evidenceDirectory) {
                std::cerr << "thoth-engine: one evidence directory per process\n";
                options.error = true;
                return options;
            }
            if (i + 1 >= argc) {
                std::cerr << "thoth-engine: --evidence-directory requires a directory path\n";
                options.error = true;
                return options;
            }
            options.evidenceDirectory = argv[++i];
            continue;
        }
        if (arg == "--index-directory") {
            if (options.indexDirectory) {
                std::cerr << "thoth-engine: one index directory per process\n";
                options.error = true;
                return options;
            }
            if (i + 1 >= argc) {
                std::cerr << "thoth-engine: --index-directory requires a directory path\n";
                options.error = true;
                return options;
            }
            options.indexDirectory = argv[++i];
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
    if (options.fixedEvidenceGoal && (options.serve || options.executePrompt)) {
        std::cerr << "thoth-engine: --fixed-evidence-goal cannot be combined with --serve or --execute\n";
        options.error = true;
    }
    if (options.indexDirectory
        && (options.serve || options.executePrompt || options.fixedEvidenceGoal)) {
        std::cerr << "thoth-engine: --index-directory cannot be combined with --serve, --execute, or --fixed-evidence-goal\n";
        options.error = true;
    }
    if (options.evidenceDirectory && !options.fixedEvidenceGoal) {
        std::cerr << "thoth-engine: --evidence-directory requires --fixed-evidence-goal\n";
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

    if (!options.executePrompt && !options.serve && !options.fixedEvidenceGoal
        && !options.indexDirectory) {
        installShutdownSignals();
    }

    sigset_t shutdown_signals {};
    if (options.serve) {
        blockShutdownSignals(shutdown_signals);
    }

    std::string fixed_goal;
    std::string resolved_evidence_dir;
    if (options.fixedEvidenceGoal) {
        std::string error;
        if (!Thoth::readFixedEvidenceGoalFile(*options.fixedEvidenceGoal, fixed_goal, error)) {
            std::cerr << "[thoth-engine] " << error << '\n';
            return 1;
        }
        if (!options.evidenceDirectory) {
            std::cerr << "[thoth-engine] fixed-evidence goal requires --evidence-directory\n";
            return 1;
        }
        if (!Thoth::resolveAuthorizedEvidenceDirectory(
                *options.evidenceDirectory, resolved_evidence_dir, error)) {
            std::cerr << "[thoth-engine] " << error << '\n';
            return 1;
        }
    }

    std::string resolved_index_dir;
    if (options.indexDirectory) {
        std::string error;
        if (!Thoth::validateWorkspaceIndexDirectory(
                *options.indexDirectory, error, resolved_index_dir)) {
            std::cerr << "[thoth-engine] " << error << '\n';
            return 1;
        }
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

        if (options.indexDirectory) {
            std::string error;
            Thoth::WorkspaceIndexResult result;
            if (!runtime->indexWorkspaceDirectory(resolved_index_dir, error, result)) {
                std::cerr << "[thoth-engine] " << error << '\n';
                runtime->shutdown();
                return 1;
            }
            std::cout << Thoth::formatWorkspaceIndexStatus(result) << '\n';
            runtime->shutdown();
            return 0;
        }

        if (options.fixedEvidenceGoal) {
            std::string error;
            std::string terminal_state;
            if (!runtime->runFixedEvidenceGoal(
                    fixed_goal, resolved_evidence_dir, error, terminal_state)) {
                std::cerr << "[thoth-engine] " << error << '\n';
                runtime->shutdown();
                return 1;
            }
            std::cout << "FIXED_EVIDENCE_GOAL " << terminal_state << '\n';
            runtime->shutdown();
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
