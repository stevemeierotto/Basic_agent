/*
 * Copyright (c) 2025 Steve Meierotto
 *
 * Thoth — headless engine entry point (Plan C)
 *
 * Licensed under the MIT License (see LICENSE in project root)
 */

#include "../include/basic_agent_plugin.h"
#include "../include/runtime_bootstrap.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <optional>
#include <string>

#ifndef THOTH_ENGINE_VERSION
#define THOTH_ENGINE_VERSION "0.2"
#endif

namespace {

std::atomic<bool> g_stop{false};

void onSignal(int) {
    g_stop = true;
}

void printUsage(std::ostream& out) {
    out << "Usage: thoth-engine [--help] [--version]\n"
        << "       thoth-engine --execute \"<prompt>\"\n"
        << "       thoth-engine\n"
        << "\n"
        << "Modes:\n"
        << "  (default)   Read prompts from stdin until EOF or SIGINT\n"
        << "  --execute   Process one prompt and exit\n"
        << "\n"
        << "Environment (see docs/GETTING_STARTED.md):\n"
        << "  THOTH_WORKSPACE_PATH, THOTH_LOGS_PATH, THOTH_INFERENCE_BASE_URL\n"
        << "  THOTH_LOG_CONFIG=1          Print resolved paths at startup\n";
}

struct CliOptions {
    bool showHelp = false;
    bool showVersion = false;
    std::optional<std::string> executePrompt;
    bool error = false;
};

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

    if (!options.executePrompt) {
        std::signal(SIGINT, onSignal);
        std::signal(SIGTERM, onSignal);
    }

    try {
        BasicAgentPlugin plugin;

        if (options.executePrompt) {
            std::cout << plugin.processInput(*options.executePrompt) << '\n';
            return 0;
        }

        std::string line;
        while (!g_stop.load() && std::getline(std::cin, line)) {
            if (line.empty()) {
                continue;
            }
            std::cout << plugin.processInput(line) << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "[thoth-engine] fatal: " << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "[thoth-engine] fatal: unknown error\n";
        return 1;
    }

    return 0;
}
